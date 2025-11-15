import easyocr
import os
import sys
from pathlib import Path

# Set UTF-8 encoding for console output
sys.stdout.reconfigure(encoding='utf-8')

# Initialize EasyOCR reader
# Using Thai language for license plates (includes numbers)
reader = easyocr.Reader(['th'])

# Directory containing license plate images
license_dir = r'C:\Users\ais84\Desktop\WORK\W2datasci\license'

# Supported image extensions
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# Get all image files from the directory
image_files = [f for f in os.listdir(license_dir)
               if os.path.splitext(f)[1].lower() in image_extensions]

print(f"Found {len(image_files)} images in {license_dir}\n")
print("=" * 80)

# Process each image
for image_file in sorted(image_files):
    image_path = os.path.join(license_dir, image_file)

    print(f"\nProcessing: {image_file}")
    print("-" * 80)

    try:
        # Read text from image
        result = reader.readtext(image_path)

        if result:
            print(f"Detected text ({len(result)} detection(s)):")
            for detection in result:
                bbox, text, confidence = detection
                print(f"  Text: '{text}' (Confidence: {confidence:.2f})")
        else:
            print("  No text detected")

    except Exception as e:
        print(f"  Error processing image: {str(e)}")

    print("-" * 80)

print("\n" + "=" * 80)
print("Processing complete!")
