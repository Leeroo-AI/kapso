# File: `tests/saving/vision_models/test_save_merge_qwen2.5vl32B_model_ocr_benchmark.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 287 |
| Functions | `format_data`, `find_lora_base_model` |
| Imports | datasets, jiwer, os, pandas, pathlib, qwen_vl_utils, sys, tests, torch, tqdm, ... +2 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive benchmark test that evaluates the OCR performance of the Qwen2.5-VL-32B vision model across different stages: base model, LoRA adapter, and merged model at various quantization levels.

**Mechanism:** The test uses OCRModelEvaluator to track Word Error Rate (WER) and Character Error Rate (CER) across configurations: (1) benchmarks the base unsloth/Qwen2.5-VL-32B-Instruct-bnb-4bit model on 200 evaluation samples, (2) applies LoRA fine-tuning with r=16 on vision and language layers for 60 training steps, (3) benchmarks the LoRA adapter model, (4) merges adapters to 16-bit using save_pretrained_merged, (5) benchmarks the merged model loaded in 16-bit, 4-bit, and 8-bit quantization modes. The find_lora_base_model helper function unwraps PeftModel layers to access the underlying base model. Results are compared via print_model_comparison to analyze accuracy differences across configurations.

**Significance:** This is a critical performance validation test for the largest Qwen2.5 vision model (32B parameters). It verifies that LoRA fine-tuning improves OCR accuracy and that model merging preserves performance across different quantization levels. The benchmark data helps validate that Unsloth's optimizations do not degrade model quality while enabling memory-efficient inference.
