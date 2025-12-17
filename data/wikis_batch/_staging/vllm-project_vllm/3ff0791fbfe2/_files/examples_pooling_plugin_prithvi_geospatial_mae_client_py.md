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

**Purpose:** Geospatial segmentation online client

**Mechanism:** This client performs online inference with the Prithvi geospatial model via vLLM's `/pooling` endpoint. It sends a GeoTIFF image URL to the server, which processes it through TerraTorch's segmentation pipeline. The response contains a base64-encoded prediction TIFF that is decoded and saved locally. Requires the model to be served with `--model-impl terratorch` and `--io-processor-plugin terratorch_segmentation`.

**Significance:** Example for using vLLM with custom I/O processors for multimodal geospatial AI. Demonstrates integration with specialized model implementations (TerraTorch) for flood segmentation from satellite imagery.
