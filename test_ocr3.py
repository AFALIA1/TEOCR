from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import os
import sys
from pathlib import Path
import time
from threading import Thread
from queue import Queue, Empty

# Set UTF-8 encoding for output
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# Timeout configuration (in seconds)
TIMEOUT_PER_IMAGE = 30  # Maximum time allowed per image

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

# Initialize the Typhoon OCR model
model_name = "scb10x/typhoon-ocr1.5-2b"
print(f"Loading model: {model_name}")

try:
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully on {device}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Process license plate images
license_dir = Path(r"C:\Users\ais84\Desktop\WORK\W2datasci\license")
image_files = sorted(license_dir.glob("*.jpg"))

print(f"\nFound {len(image_files)} images in license directory\n")
print("=" * 80)

# Create output file for results
output_file = Path(r"C:\Users\ais84\Desktop\WORK\W2datasci\ocr_results.txt")
results = []

def perform_ocr(img_path, processor, model, device):
    """Perform OCR on a single image."""
    # Prepare the message with image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(img_path)},
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

# Track total processing time
total_start_time = time.time()

for img_path in image_files:
    img_start_time = time.time()
    try:
        # Run OCR with timeout
        output, error = process_image_with_timeout(
            perform_ocr,
            (img_path, processor, model, device),
            TIMEOUT_PER_IMAGE
        )

        img_elapsed = time.time() - img_start_time

        if error:
            # Timeout or other error occurred
            error_text = f"{img_path.name}: ERROR - {error} (time: {img_elapsed:.2f}s)"
            results.append(error_text)
            print(f"\nFile: {img_path.name}")
            print(f"ERROR: {error}")
            print(f"Time: {img_elapsed:.2f}s")
            print("-" * 80)
        else:
            # Success
            result_text = f"{img_path.name}: {output} (time: {img_elapsed:.2f}s)"
            results.append(result_text)

            print(f"\nFile: {img_path.name}")
            print(f"OCR Result: {output}")
            print(f"Time: {img_elapsed:.2f}s")
            print("-" * 80)

    except Exception as e:
        img_elapsed = time.time() - img_start_time
        error_text = f"{img_path.name}: ERROR - {str(e)} (time: {img_elapsed:.2f}s)"
        results.append(error_text)
        print(f"\nError processing {img_path.name}: {e}")
        print(f"Time: {img_elapsed:.2f}s")
        print("-" * 80)

# Calculate total time
total_elapsed = time.time() - total_start_time

# Save all results to file
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("License Plate OCR Results\n")
    f.write("=" * 80 + "\n\n")
    for result in results:
        f.write(result + "\n")
    f.write("\n" + "=" * 80 + "\n")
    f.write(f"Total processing time: {total_elapsed:.2f}s\n")
    f.write(f"Average time per image: {total_elapsed/len(image_files):.2f}s\n")
    f.write(f"Timeout limit per image: {TIMEOUT_PER_IMAGE}s\n")

print(f"\nOCR processing complete!")
print(f"Total time: {total_elapsed:.2f}s")
print(f"Average time per image: {total_elapsed/len(image_files):.2f}s" if image_files else "No images processed")
print(f"Results saved to: {output_file}")
