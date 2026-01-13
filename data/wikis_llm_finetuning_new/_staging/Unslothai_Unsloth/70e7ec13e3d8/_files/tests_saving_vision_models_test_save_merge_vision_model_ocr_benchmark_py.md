# File: `tests/saving/vision_models/test_save_merge_vision_model_ocr_benchmark.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 287 |
| Functions | `format_data`, `find_lora_base_model` |
| Imports | datasets, jiwer, os, pandas, pathlib, qwen_vl_utils, sys, tests, torch, tqdm, ... +2 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmark test for evaluating OCR performance of the Qwen2-VL-7B vision model through the complete fine-tuning, merging, and inference pipeline with performance comparisons across model configurations.

**Mechanism:** The test follows the same structure as the 32B benchmark but uses Qwen2-VL-7B-Instruct: (1) loads the French OCR dataset and formats into multi-modal messages, (2) creates an OCRModelEvaluator instance for WER/CER tracking, (3) benchmarks the base 4-bit quantized model, (4) applies LoRA with finetune_vision_layers and finetune_language_layers for 60 steps, (5) benchmarks the LoRA adapter model, (6) merges to 16-bit and saves locally, (7) reloads and benchmarks the merged model at 16-bit, 4-bit, and 8-bit precisions. The find_lora_base_model helper traverses base_model and model attributes to extract the underlying Qwen2 model from PeftModel wrapper. Final comparison report shows accuracy progression across configurations.

**Significance:** This test serves as the primary regression test for vision model save/merge functionality with the mid-sized 7B model. It validates that LoRA adapter training improves OCR accuracy, merged models retain quality, and quantization trade-offs are acceptable. The 7B model is more accessible for CI/testing than the 32B variant while still thoroughly testing the vision model workflow.
