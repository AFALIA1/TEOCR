"""
Example script for testing the License Plate OCR API
"""
import requests
import base64
from pathlib import Path

# API endpoint
API_URL = "http://localhost:8000"


def test_health():
    """Test if the API is healthy"""
    response = requests.get(f"{API_URL}/health")
    print("Health Check:", response.json())
    return response.status_code == 200


def test_single_image(image_path):
    """
    Test OCR on a single image

    Args:
        image_path: Path to the image file
    """
    with open(image_path, 'rb') as f:
        files = {'file': (Path(image_path).name, f, 'image/jpeg')}
        response = requests.post(f"{API_URL}/ocr", files=files)

    result = response.json()
    print(f"\nSingle Image OCR Result:")
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Text: {result['text']}")
        print(f"Processing Time: {result['processing_time']:.2f}s")
    else:
        print(f"Error: {result['error']}")

    return result


def test_batch_images(image_paths):
    """
    Test OCR on multiple images

    Args:
        image_paths: List of paths to image files
    """
    files = []
    for img_path in image_paths:
        with open(img_path, 'rb') as f:
            files.append(('files', (Path(img_path).name, f.read(), 'image/jpeg')))

    response = requests.post(f"{API_URL}/ocr/batch", files=files)
    result = response.json()

    print(f"\nBatch OCR Results:")
    print(f"Total Images: {result['total_images']}")
    print(f"Successful: {result['successful']}")
    print(f"Failed: {result['failed']}")

    for i, res in enumerate(result['results']):
        print(f"\nImage {i+1}:")
        print(f"  Success: {res['success']}")
        if res['success']:
            print(f"  Text: {res['text']}")
            print(f"  Processing Time: {res['processing_time']:.2f}s")
        else:
            print(f"  Error: {res['error']}")

    return result


def test_base64_image(image_path):
    """
    Test OCR using base64-encoded image

    Args:
        image_path: Path to the image file
    """
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    response = requests.post(
        f"{API_URL}/ocr/base64",
        json={"image_base64": image_data}
    )

    result = response.json()
    print(f"\nBase64 Image OCR Result:")
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Text: {result['text']}")
        print(f"Processing Time: {result['processing_time']:.2f}s")
    else:
        print(f"Error: {result['error']}")

    return result


def main():
    """Main test function"""
    print("Testing License Plate OCR API\n")
    print("=" * 80)

    # Test health
    if not test_health():
        print("API is not healthy. Please start the server first.")
        return

    # Example: Test with a single image
    # Uncomment and modify the path to your image
    # image_path = r"C:\path\to\your\license_plate.jpg"
    # test_single_image(image_path)

    # Example: Test with multiple images
    # image_paths = [
    #     r"C:\path\to\image1.jpg",
    #     r"C:\path\to\image2.jpg",
    # ]
    # test_batch_images(image_paths)

    # Example: Test with base64 image
    # test_base64_image(image_path)

    print("\n" + "=" * 80)
    print("Testing complete!")


if __name__ == "__main__":
    main()
