# License Plate OCR API

FastAPI-based OCR service for extracting text from license plate images using the Typhoon OCR model (scb10x/typhoon-ocr1.5-2b).

## Features

- Single image OCR
- Batch image processing
- Base64 image support
- Timeout handling (30 seconds per image)
- GPU acceleration (if available)
- RESTful API with automatic documentation

## Requirements

Install dependencies:
```bash
pip install fastapi uvicorn transformers torch pillow qwen-vl-utils
```

## Starting the API Server

Run the server using:

```bash
python license.py
```

Or with custom settings:
```bash
uvicorn license:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: `http://localhost:8000`

## API Endpoints

### 1. Root Endpoint
**GET** `/`

Returns API information and available endpoints.

```bash
curl http://localhost:8000/
```

### 2. Health Check
**GET** `/health`

Check if the model is loaded and API is ready.

```bash
curl http://localhost:8000/health
```

### 3. Single Image OCR
**POST** `/ocr`

Extract text from a single image file.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Image file

**Example:**
```bash
curl -X POST "http://localhost:8000/ocr" \
  -F "file=@/path/to/license_plate.jpg"
```

**Python example:**
```python
import requests

with open('license_plate.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/ocr', files=files)
    result = response.json()
    print(result['text'])
```

**Response:**
```json
{
  "success": true,
  "text": "ABC-1234",
  "processing_time": 1.23
}
```

### 4. Batch Image OCR
**POST** `/ocr/batch`

Extract text from multiple images at once.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Multiple image files

**Example:**
```bash
curl -X POST "http://localhost:8000/ocr/batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

**Python example:**
```python
import requests

files = [
    ('files', open('image1.jpg', 'rb')),
    ('files', open('image2.jpg', 'rb'))
]
response = requests.post('http://localhost:8000/ocr/batch', files=files)
results = response.json()
```

**Response:**
```json
{
  "results": [
    {
      "success": true,
      "text": "ABC-1234",
      "processing_time": 1.23
    },
    {
      "success": true,
      "text": "XYZ-5678",
      "processing_time": 1.45
    }
  ],
  "total_images": 2,
  "successful": 2,
  "failed": 0
}
```

### 5. Base64 Image OCR
**POST** `/ocr/base64`

Extract text from a base64-encoded image.

**Request:**
- Content-Type: `application/json`
- Body: JSON with base64-encoded image

**Python example:**
```python
import requests
import base64

with open('license_plate.jpg', 'rb') as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

response = requests.post(
    'http://localhost:8000/ocr/base64',
    json={'image_base64': image_base64}
)
result = response.json()
```

## Interactive API Documentation

FastAPI provides automatic interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Configuration

### Timeout Settings

You can modify the timeout in `license.py`:
```python
TIMEOUT_PER_IMAGE = 30  # seconds
```

### Model Settings

The API uses the Typhoon OCR model by default:
```python
model_name = "scb10x/typhoon-ocr1.5-2b"
```

### Server Settings

Modify the server settings at the bottom of `license.py`:
```python
uvicorn.run(
    app,
    host="0.0.0.0",  # Change to "127.0.0.1" for local only
    port=8000,       # Change port if needed
    log_level="info"
)
```

## Error Handling

The API returns appropriate error messages:

- **503 Service Unavailable**: Model not loaded
- **500 Internal Server Error**: Processing error
- **Timeout errors**: Included in response with error message

Example error response:
```json
{
  "success": false,
  "error": "TIMEOUT after 30 seconds",
  "processing_time": 30.01
}
```

## Testing

Use the provided test script:

```bash
python test_api.py
```

Modify the script to point to your test images.

## Performance

- **GPU**: Recommended for faster processing
- **CPU**: Supported but slower
- **Timeout**: 30 seconds per image (configurable)
- **Batch processing**: Processes images sequentially

## Model Information

- **Model**: scb10x/typhoon-ocr1.5-2b
- **Type**: Vision-to-Sequence (AutoModelForVision2Seq)
- **Framework**: Transformers + PyTorch
- **Precision**: bfloat16 (GPU) / float32 (CPU)
