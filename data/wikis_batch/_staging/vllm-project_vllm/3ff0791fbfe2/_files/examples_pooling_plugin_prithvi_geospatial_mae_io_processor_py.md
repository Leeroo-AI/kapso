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

**Purpose:** Geospatial segmentation offline inference

**Mechanism:** This script performs offline inference with the Prithvi geospatial model using vLLM's LLM class. It loads the model with TerraTorch implementation, processes GeoTIFF images through the I/O processor plugin, and outputs segmentation masks. The script uses `llm.encode()` with `pooling_task="plugin"` to generate predictions, then decodes and saves the base64-encoded output as a TIFF file.

**Significance:** Example for offline batch processing of geospatial data with custom model implementations. Shows how to configure vLLM with specialized plugins (`io_processor_plugin`, `model_impl`) and multimodal embeddings for non-text AI tasks.
