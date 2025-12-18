# File: `examples/pooling/plugin/prithvi_geospatial_mae_io_processor.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 58 |
| Functions | `main` |
| Imports | base64, os, torch, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Offline inference example for geospatial imagery

**Mechanism:** Uses vLLM's `LLM` class directly (not via server) to process geospatial TIFF images with the Prithvi-EO-2.0 model. Configures the model with TerraTorch integration, sets float16 dtype, encodes images with the `plugin` pooling task, decodes base64 output, and saves predictions. Requires `terratorch>=v1.1` installation.

**Significance:** Demonstrates offline/batch processing of geospatial imagery without requiring an API server. Shows how to configure vLLM for specialized multimodal models with custom IO processors for Earth observation tasks like flood detection and land cover classification.
