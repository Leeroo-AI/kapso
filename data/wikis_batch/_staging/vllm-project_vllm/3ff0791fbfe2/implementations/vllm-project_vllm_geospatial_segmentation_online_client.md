# Prithvi Geospatial Segmentation - Online Client

**Source:** `examples/pooling/plugin/prithvi_geospatial_mae_client.py`
**Repository:** [vllm-project/vllm](https://github.com/vllm-project/vllm)
**Lines:** 56

## Overview

This example demonstrates online inference for geospatial segmentation using the Prithvi Earth Observation model through vLLM's `/pooling` endpoint. It shows how to process GeoTIFF satellite imagery for tasks like flood detection by sending images to a vLLM server configured with custom I/O processors and model implementations.

## Implementation Pattern

### Architecture Design

**Client Side:**
- Constructs request with image URL and processing parameters
- Posts to `/pooling` endpoint with plugin task type
- Receives base64-encoded segmentation mask
- Decodes and saves prediction as GeoTIFF

**Server Side (vLLM):**
- Loads Prithvi model with TerraTorch implementation
- Processes GeoTIFF through specialized I/O processor
- Generates segmentation predictions
- Returns results as base64-encoded TIFF

### Use Case: Flood Segmentation

The example processes satellite imagery to identify flooded areas:

**Input:** Multi-band GeoTIFF from satellite observation
**Output:** Binary segmentation mask (flooded vs. non-flooded pixels)
**Application:** Disaster response, urban planning, environmental monitoring

## Technical Implementation

### 1. Request Construction

```python
def main():
    image_url = "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/valencia_example_2024-10-26.tiff"  # noqa: E501
    server_endpoint = "http://localhost:8000/pooling"

    request_payload_url = {
        "data": {
            "data": image_url,
            "data_format": "url",
            "image_format": "tiff",
            "out_data_format": "b64_json",
        },
        "priority": 0,
        "model": "christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM",
    }

    ret = requests.post(server_endpoint, json=request_payload_url)
```

**Request Structure:**

**data (nested dict):**
Configuration for input/output data handling.

**data.data:**
- URL to the input GeoTIFF image
- Can be HTTP/HTTPS URL or HuggingFace repo path
- Must be publicly accessible by the server

**data.data_format:**
- `"url"`: Fetch image from URL
- Alternative: `"base64"` for embedded image data

**data.image_format:**
- `"tiff"`: GeoTIFF format
- Supports multi-band satellite imagery
- Preserves geospatial metadata

**data.out_data_format:**
- `"b64_json"`: Return as base64-encoded JSON
- Enables transport over HTTP
- Preserves binary TIFF structure

**priority:**
- Request priority (0 = default)
- Higher numbers = higher priority
- Useful for time-sensitive disaster response

**model:**
- Full model identifier
- Must match server configuration
- Points to Prithvi geospatial model

### 2. Response Processing

```python
print(f"response.status_code: {ret.status_code}")
print(f"response.reason:{ret.reason}")

response = ret.json()

decoded_image = base64.b64decode(response["data"]["data"])

out_path = os.path.join(os.getcwd(), "online_prediction.tiff")

with open(out_path, "wb") as f:
    f.write(decoded_image)
```

**Response Structure:**
```json
{
  "data": {
    "data": "base64_encoded_tiff_data...",
    "format": "tiff"
  },
  "model": "christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM",
  "processing_time": 1.234
}
```

**Processing Steps:**
1. Check HTTP status code for success (200)
2. Parse JSON response
3. Extract base64-encoded TIFF from `response["data"]["data"]`
4. Decode base64 to binary TIFF data
5. Write to file with `.tiff` extension

## Server Configuration

### Starting the vLLM Server

The example requires specific server configuration:

```bash
vllm serve christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM \
  --model-impl terratorch \
  --trust-remote-code \
  --skip-tokenizer-init \
  --enforce-eager \
  --io-processor-plugin terratorch_segmentation \
  --enable-mm-embeds
```

**Configuration Flags:**

**--model-impl terratorch:**
Uses TerraTorch model implementation instead of default transformers.
- TerraTorch: Framework for geospatial foundation models
- Provides specialized layers for satellite imagery

**--trust-remote-code:**
Allows execution of custom code in the model repository.
- Required for models with specialized processing
- Review code before enabling in production

**--skip-tokenizer-init:**
Bypasses text tokenizer initialization.
- Not needed for image-only models
- Reduces startup time and memory

**--enforce-eager:**
Disables graph compilation for faster startup.
- Trades some inference speed for quicker initialization
- Recommended for development and testing

**--io-processor-plugin terratorch_segmentation:**
Loads the segmentation I/O processor plugin.
- Handles GeoTIFF input processing
- Formats segmentation output as GeoTIFF
- Manages geospatial metadata

**--enable-mm-embeds:**
Enables multimodal embedding support.
- Required for processing non-text inputs
- Allows image data to flow through the model

## Dependencies

### Client Requirements

```python
import base64
import os
import requests
```

**Installation:**
```bash
pip install requests
```

No special dependencies on client side - standard library and requests only.

### Server Requirements

**TerraTorch:**
```bash
pip install terratorch>=v1.1
```

Essential for Prithvi model support:
- Geospatial data loaders
- Specialized model architectures
- Segmentation post-processing

**vLLM:**
```bash
pip install vllm
```

With plugin support for custom I/O processors.

**Additional Dependencies:**
- GDAL for GeoTIFF handling
- PyTorch with CUDA support
- NumPy for array operations

## GeoTIFF Processing

### Input Image Characteristics

**Prithvi Model Expectations:**

**Bands:**
Multi-spectral satellite imagery (typically 6 bands):
- Blue, Green, Red (visible spectrum)
- Near-infrared (NIR)
- Short-wave infrared (SWIR)

**Resolution:**
30m per pixel (Landsat-like resolution)

**Format:**
GeoTIFF with geospatial metadata:
- Coordinate reference system (CRS)
- Geotransform (pixel-to-coordinate mapping)
- Band descriptions

**Example Image:**
The Valencia flood example (2024-10-26) shows flooding from recent storm events.

### Output Segmentation Mask

**Format:**
Single-band GeoTIFF with integer class labels:
- 0: Non-flooded areas
- 1: Flooded areas

**Geospatial Metadata:**
Preserved from input:
- Same CRS as input image
- Same geotransform
- Can be overlaid in GIS software

**Visualization:**
```python
import rasterio
from matplotlib import pyplot as plt

with rasterio.open("online_prediction.tiff") as src:
    prediction = src.read(1)

plt.imshow(prediction, cmap='Blues')
plt.title("Flood Segmentation")
plt.colorbar(label="Flooded (1) / Not Flooded (0)")
plt.show()
```

## Production Patterns

### Error Handling

```python
def process_geospatial_image(image_url, server_endpoint, model_name):
    """Process geospatial image with error handling."""
    request_payload = {
        "data": {
            "data": image_url,
            "data_format": "url",
            "image_format": "tiff",
            "out_data_format": "b64_json",
        },
        "priority": 0,
        "model": model_name,
    }

    try:
        response = requests.post(
            server_endpoint,
            json=request_payload,
            timeout=120  # Segmentation can take time
        )
        response.raise_for_status()

        result = response.json()

        # Validate response structure
        if "data" not in result or "data" not in result["data"]:
            raise ValueError("Invalid response format")

        decoded_image = base64.b64decode(result["data"]["data"])

        return decoded_image

    except requests.exceptions.Timeout:
        print("Request timed out - image may be too large")
        raise
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e}")
        print(f"Response: {e.response.text}")
        raise
    except base64.binascii.Error as e:
        print(f"Base64 decode error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
```

### Batch Processing

```python
import concurrent.futures

def process_images_parallel(image_urls, server_endpoint, model_name, max_workers=4):
    """Process multiple images in parallel."""
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(
                process_geospatial_image,
                url,
                server_endpoint,
                model_name
            ): url
            for url in image_urls
        }

        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                data = future.result()
                results.append({"url": url, "data": data, "success": True})
            except Exception as e:
                print(f"Failed to process {url}: {e}")
                results.append({"url": url, "error": str(e), "success": False})

    return results
```

### Local File Upload

For images not accessible via URL:

```python
def process_local_image(image_path, server_endpoint, model_name):
    """Process local GeoTIFF file."""
    with open(image_path, "rb") as f:
        image_data = f.read()

    # Encode as base64
    b64_image = base64.b64encode(image_data).decode('utf-8')

    request_payload = {
        "data": {
            "data": b64_image,
            "data_format": "base64",  # Changed from "url"
            "image_format": "tiff",
            "out_data_format": "b64_json",
        },
        "priority": 0,
        "model": model_name,
    }

    response = requests.post(server_endpoint, json=request_payload, timeout=120)
    response.raise_for_status()

    result = response.json()
    return base64.b64decode(result["data"]["data"])
```

### Result Validation

```python
import rasterio

def validate_output(output_path, expected_shape=None):
    """Validate output segmentation mask."""
    try:
        with rasterio.open(output_path) as src:
            # Check it's a valid GeoTIFF
            assert src.count >= 1, "No bands in output"

            # Read prediction
            prediction = src.read(1)

            # Check shape if provided
            if expected_shape:
                assert prediction.shape == expected_shape, \
                    f"Shape mismatch: {prediction.shape} != {expected_shape}"

            # Check values are valid class labels
            unique_values = np.unique(prediction)
            assert all(v >= 0 for v in unique_values), \
                "Negative class labels found"

            # Check geospatial metadata
            assert src.crs is not None, "Missing CRS"
            assert src.transform is not None, "Missing geotransform"

            return True

    except Exception as e:
        print(f"Validation failed: {e}")
        return False
```

## Integration Examples

### Flask Web Service

```python
from flask import Flask, request, jsonify, send_file
import tempfile
import os

app = Flask(__name__)

VLLM_ENDPOINT = "http://localhost:8000/pooling"
MODEL_NAME = "christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM"

@app.route('/segment', methods=['POST'])
def segment_image():
    """Segment geospatial image."""
    data = request.json
    image_url = data.get('image_url')

    if not image_url:
        return jsonify({"error": "image_url required"}), 400

    try:
        # Process image
        decoded_image = process_geospatial_image(
            image_url,
            VLLM_ENDPOINT,
            MODEL_NAME
        )

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tiff') as tmp:
            tmp.write(decoded_image)
            tmp_path = tmp.name

        # Return file
        return send_file(
            tmp_path,
            mimetype='image/tiff',
            as_attachment=True,
            download_name='segmentation.tiff'
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
```

### Disaster Response Pipeline

```python
class FloodMonitoringSystem:
    """Automated flood monitoring using satellite imagery."""

    def __init__(self, vllm_endpoint, model_name):
        self.vllm_endpoint = vllm_endpoint
        self.model_name = model_name

    def process_satellite_pass(self, image_url, region_name):
        """Process new satellite imagery for flood detection."""
        print(f"Processing {region_name}...")

        # Get segmentation
        prediction_data = process_geospatial_image(
            image_url,
            self.vllm_endpoint,
            self.model_name
        )

        # Save prediction
        output_path = f"floods_{region_name}_{datetime.now().isoformat()}.tiff"
        with open(output_path, "wb") as f:
            f.write(prediction_data)

        # Analyze results
        flooded_area = self.calculate_flooded_area(output_path)

        # Alert if significant flooding
        if flooded_area > 10.0:  # km²
            self.send_alert(region_name, flooded_area, output_path)

        return flooded_area

    def calculate_flooded_area(self, prediction_path):
        """Calculate flooded area in square kilometers."""
        with rasterio.open(prediction_path) as src:
            prediction = src.read(1)
            transform = src.transform

            # Count flooded pixels
            flooded_pixels = (prediction == 1).sum()

            # Calculate pixel area in km²
            pixel_width = transform[0]  # degrees
            pixel_height = -transform[4]  # degrees (negative)

            # Approximate area (works for small regions)
            pixel_area_km2 = (pixel_width * 111) * (pixel_height * 111)

            return flooded_pixels * pixel_area_km2

    def send_alert(self, region_name, area_km2, prediction_path):
        """Send alert for significant flooding."""
        print(f"ALERT: {area_km2:.2f} km² flooding detected in {region_name}")
        # Integrate with notification system
        # e.g., send email, SMS, push notification
```

## Performance Considerations

### Timeout Configuration

**Typical Processing Times:**
- Small image (512x512): 2-5 seconds
- Medium image (1024x1024): 5-15 seconds
- Large image (2048x2048): 15-60 seconds

**Recommended Timeouts:**
```python
requests.post(..., timeout=120)  # 2 minutes
```

Adjust based on image size and server hardware.

### Request Priority

For disaster response scenarios:

```python
# High priority for urgent areas
urgent_request = {
    "data": {...},
    "priority": 10,  # Higher priority
    "model": model_name,
}

# Low priority for routine monitoring
routine_request = {
    "data": {...},
    "priority": 0,  # Default priority
    "model": model_name,
}
```

Server processes high-priority requests first.

### Caching Strategies

```python
import hashlib

def get_cache_key(image_url):
    """Generate cache key for image URL."""
    return hashlib.md5(image_url.encode()).hexdigest()

def process_with_cache(image_url, cache_dir):
    """Process image with file-based caching."""
    cache_key = get_cache_key(image_url)
    cache_path = os.path.join(cache_dir, f"{cache_key}.tiff")

    # Check cache
    if os.path.exists(cache_path):
        print(f"Cache hit for {image_url}")
        with open(cache_path, "rb") as f:
            return f.read()

    # Process image
    result = process_geospatial_image(image_url, ...)

    # Save to cache
    with open(cache_path, "wb") as f:
        f.write(result)

    return result
```

## Troubleshooting

### Common Issues

**Invalid Image URL:**
```
requests.exceptions.HTTPError: 404 Not Found
```

**Solution:**
- Verify URL is accessible: `curl -I <url>`
- Check for authentication requirements
- Ensure URL points to TIFF file

**Decode Error:**
```
binascii.Error: Incorrect padding
```

**Solution:**
- Verify response contains valid base64
- Check response structure matches expectations
- Inspect raw response: `print(response.text)`

**Server Not Configured:**
```
{"error": "Model not found"}
```

**Solution:**
- Confirm server started with correct model
- Check `--model-impl terratorch` flag
- Verify `--io-processor-plugin terratorch_segmentation` flag

**Memory Errors:**
```
torch.cuda.OutOfMemoryError
```

**Solution:**
- Reduce image size before processing
- Tile large images into smaller patches
- Increase GPU memory or use smaller batch size

## Related Examples

- **prithvi_geospatial_mae_io_processor.py:** Offline inference version
- **openai_pooling_client.py:** Generic pooling endpoint example

## References

- **Prithvi Model:** [HuggingFace](https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM)
- **TerraTorch:** [Documentation](https://github.com/IBM/terratorch)
- **GeoTIFF Format:** GDAL documentation
- **Flood Detection:** Remote sensing applications
