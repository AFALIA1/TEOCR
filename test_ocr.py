import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import sys
import io
import time
import gc

# Fix Windows console encoding for Thai text
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Enable torch optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Load model and processor
model_name = "scb10x/typhoon-ocr-7b"
print("Loading model...")

# Load processor with optimizations
processor = AutoProcessor.from_pretrained(
    model_name,
    trust_remote_code=True
)

# Configure 4-bit quantization (Q4) with optimizations
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

# Set model to eval mode for inference optimizations
model.eval()

print("Model loaded successfully!")

# Load and process image
start = time.time()

image_path = "A1.png"
print(f"\nProcessing image: {image_path}")
image = Image.open(image_path).convert("RGB")

# Preprocess image with text prompt for OCR
prompt = "<|vision_start|><|image_pad|><|vision_end|>Extract all text from this image."
inputs = processor(text=prompt, images=image, return_tensors="pt")

# Move inputs to device efficiently
inputs = {k: v.to(model.device, non_blocking=True) for k, v in inputs.items()}

# Generate text with optimized settings
print("Running OCR...")
with torch.inference_mode():  # More efficient than no_grad()
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=False,  # Deterministic greedy decoding is faster
        use_cache=True,  # Enable KV cache for faster generation
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        num_beams=1  # Greedy decoding, no beam search
    )

# Decode text
text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

# Clean up
del inputs, outputs
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

end = time.time()

print(f"Processing time: {end - start:.2f} seconds")
print("\n" + "="*50)
print("OCR Result:")
print("="*50)
print(text)
print("="*50)
print(gc.get_stats())
