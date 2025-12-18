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

**Purpose:** Advanced geospatial image segmentation pipeline for flood detection

**Mechanism:** Comprehensive workflow for Sen1Floods11 dataset: (1) loads and preprocesses multi-spectral satellite imagery with albumentations transforms, (2) extracts temporal/spatial coordinates from filenames, (3) applies sliding window approach for large images, (4) runs Prithvi-EO-2.0 model inference through vLLM with TerraTorch integration, (5) post-processes predictions with interpolation, and (6) generates georeferenced TIFF outputs with RGB overlays. Includes extensive GeoTIFF handling with rasterio.

**Significance:** Production-ready example for satellite image segmentation showing real-world Earth observation workflows. Demonstrates handling of multi-temporal, multi-spectral geospatial data at scale with proper coordinate systems, tiling strategies for memory efficiency, and scientific data visualization. Critical for disaster response and environmental monitoring applications.
