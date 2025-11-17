from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import io
import sys
from threading import Thread
from queue import Queue, Empty
from typing import List, Dict, Any
import base64
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Set UTF-8 encoding for output
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# Timeout configuration (in seconds)
TIMEOUT_PER_IMAGE = 30  # Maximum time allowed per image

# Global model variables
model = None
processor = None
device = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for model loading and cleanup"""
    global model, processor, device

    # Startup: Load the model
    model_name = "scb10x/typhoon-ocr1.5-2b"
    print(f"Loading model: {model_name}")

    try:
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        # Move model to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        print(f"Model loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

    yield

    # Shutdown: Cleanup if needed
    print("Shutting down...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="License Plate OCR API",
    description="API for extracting text from license plate images using Typhoon OCR",
    version="1.0.0",
    lifespan=lifespan
)


class OCRResponse(BaseModel):
    """Response model for OCR results"""
    success: bool
    text: str = None
    error: str = None
    processing_time: float = None


class BatchOCRResponse(BaseModel):
    """Response model for batch OCR results"""
    results: List[OCRResponse]
    total_images: int
    successful: int
    failed: int


def process_image_with_timeout(func, args, timeout):
    """Run a function with a timeout limit."""
    result_queue = Queue()

    def worker():
        try:
            result = func(*args)
            result_queue.put(('success', result))
        except Exception as e:
            result_queue.put(('error', e))

    thread = Thread(target=worker)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        return None, f"TIMEOUT after {timeout} seconds"

    try:
        status, result = result_queue.get_nowait()
        if status == 'success':
            return result, None
        else:
            return None, str(result)
    except Empty:
        return None, "Unknown error"


def perform_ocr(image: Image.Image, processor, model, device):
    """
    Perform OCR on a single image.

    Args:
        image: PIL Image object
        processor: AutoProcessor for the model
        model: AutoModelForVision2Seq model
        device: Device to run inference on

    Returns:
        str: Extracted text from the image
    """
    # Prepare the message with image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Extract all text from this image."}
            ]
        }
    ]

    # Prepare inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    # Perform OCR
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

    return output


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "License Plate OCR API",
        "version": "1.0.0",
        "model": "scb10x/typhoon-ocr1.5-2b",
        "device": device,
        "endpoints": {
            "/ocr": "POST - Extract text from a single image",
            "/ocr/batch": "POST - Extract text from multiple images",
            "/health": "GET - Check API health status"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "status": "healthy",
        "model_loaded": True,
        "device": device
    }


@app.post("/ocr", response_model=OCRResponse)
async def extract_text(file: UploadFile = File(...)):
    """
    Extract text from a single license plate image

    Args:
        file: Image file (JPG, PNG, etc.)

    Returns:
        OCRResponse with extracted text or error
    """
    import time

    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Perform OCR with timeout
        start_time = time.time()
        output, error = process_image_with_timeout(
            perform_ocr,
            (image, processor, model, device),
            TIMEOUT_PER_IMAGE
        )
        processing_time = time.time() - start_time

        if error:
            return OCRResponse(
                success=False,
                error=error,
                processing_time=processing_time
            )

        return OCRResponse(
            success=True,
            text=output,
            processing_time=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/ocr/batch", response_model=BatchOCRResponse)
async def extract_text_batch(files: List[UploadFile] = File(...)):
    """
    Extract text from multiple license plate images

    Args:
        files: List of image files

    Returns:
        BatchOCRResponse with results for all images
    """
    import time

    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    successful = 0
    failed = 0

    for file in files:
        try:
            # Read and validate image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))

            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Perform OCR with timeout
            start_time = time.time()
            output, error = process_image_with_timeout(
                perform_ocr,
                (image, processor, model, device),
                TIMEOUT_PER_IMAGE
            )
            processing_time = time.time() - start_time

            if error:
                results.append(OCRResponse(
                    success=False,
                    error=error,
                    processing_time=processing_time
                ))
                failed += 1
            else:
                results.append(OCRResponse(
                    success=True,
                    text=output,
                    processing_time=processing_time
                ))
                successful += 1

        except Exception as e:
            results.append(OCRResponse(
                success=False,
                error=f"Error processing {file.filename}: {str(e)}"
            ))
            failed += 1

    return BatchOCRResponse(
        results=results,
        total_images=len(files),
        successful=successful,
        failed=failed
    )


@app.post("/ocr/base64")
async def extract_text_base64(image_base64: str):
    """
    Extract text from a base64-encoded image

    Args:
        image_base64: Base64-encoded image string

    Returns:
        OCRResponse with extracted text or error
    """
    import time

    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Perform OCR with timeout
        start_time = time.time()
        output, error = process_image_with_timeout(
            perform_ocr,
            (image, processor, model, device),
            TIMEOUT_PER_IMAGE
        )
        processing_time = time.time() - start_time

        if error:
            return OCRResponse(
                success=False,
                error=error,
                processing_time=processing_time
            )

        return OCRResponse(
            success=True,
            text=output,
            processing_time=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # Run the API server
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8080,
        log_level="info"
    )
