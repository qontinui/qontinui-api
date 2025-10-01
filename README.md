# Qontinui API Service

REST API service that exposes real qontinui library image recognition capabilities for web-based testing.

## Features

- **Real Qontinui Pattern Matching**: Uses actual qontinui Find operations
- **State Detection**: Detect which states are present in screenshots
- **Location Validation**: Validate locations with image-relative positioning
- **Web-Friendly**: Base64 image support for easy browser integration

## Endpoints

### Vision Operations

#### POST /find
Find a single template match in a screenshot using qontinui's real pattern matching.

```json
{
  "screenshot": "base64_image_data",
  "template": "base64_image_data",
  "similarity": 0.8,
  "search_region": {"x": 0, "y": 0, "width": 100, "height": 100}
}
```

#### POST /find_all
Find all template matches in a screenshot.

#### POST /detect_states
Detect which states from a list are present in a screenshot.

```json
{
  "screenshot": "base64_image_data",
  "states": [...],
  "similarity": 0.8
}
```

#### POST /validate_location
Validate if a location is accessible, with optional image-relative positioning.

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Start the service:
```bash
./start.sh
```

Or manually:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Documentation

When running, interactive API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Integration with Frontend

The API accepts base64-encoded images and returns JSON responses with match regions, scores, and detection results. Perfect for integration with the qontinui-web frontend for visual testing and validation.
