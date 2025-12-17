# File: `examples/pooling/plugin/prithvi_geospatial_mae_offline.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 419 |
| Classes | `PrithviMAE` |
| Functions | `generate_datamodule`, `process_channel_group`, `read_geotiff`, `save_geotiff`, `load_example`, `run_model`, `main` |
| Imports | albumentations, argparse, datetime, einops, numpy, os, rasterio, regex, terratorch, torch, ... +1 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Complex geospatial flood segmentation

**Mechanism:** Comprehensive flood segmentation pipeline using Prithvi geospatial model via vLLM. The script: (1) loads GeoTIFF satellite images with temporal/location metadata, (2) applies standardization and windowing for large images, (3) runs inference on sliding windows through vLLM's encode API with plugin pooling task, (4) reconstructs full segmentation from patches, and (5) generates visualization outputs overlaying predictions on RGB imagery. Integrates with TerraTorch's Sen1Floods11 datamodule for preprocessing.

**Significance:** Advanced example demonstrating production-grade geospatial AI workflows with vLLM. Shows handling of large satellite images, sliding window inference, geospatial coordinate integration, and visualization pipelines for flood detection applications.
