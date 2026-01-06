# File: `examples/pooling/plugin/prithvi_geospatial_mae_client.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 56 |
| Functions | `main` |
| Imports | base64, os, requests |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Online inference client for geospatial image processing

**Mechanism:** Makes HTTP POST requests to vLLM's `/pooling` endpoint with geospatial TIFF images. Sends a geotiff URL to the Prithvi-EO-2.0 model for processing, receives base64-encoded TIFF output, decodes it, and saves locally. Requires TerraTorch and specific vLLM server configuration with `--model-impl terratorch` and `--io-processor-plugin terratorch_segmentation`.

**Significance:** Demonstrates multimodal inference for geospatial/Earth observation imagery, showing how vLLM can process specialized scientific data formats (GeoTIFF) through plugin-based IO processors. Important for remote sensing and satellite image analysis workflows.
