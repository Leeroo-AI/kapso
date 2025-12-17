# File: `examples/offline_inference/mistral-small.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 186 |
| Functions | `run_simple_demo`, `run_advanced_demo`, `parse_args`, `main` |
| Imports | argparse, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates Mistral-Small-3.1-24B-Instruct multimodal inference

**Mechanism:** Provides simple and advanced demo modes using Mistral-Small model with images. Supports both mistral and HF formats. Configures mm_processor_cache_gb, limit_mm_per_prompt, uses ImageAsset and URL-based images. Advanced demo handles multiple images per message with different chat history.

**Significance:** Example showcasing Mistral-Small vision-language model usage with vLLM including cache optimization.
