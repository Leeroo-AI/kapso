# File: `tests/saving/vision_models/test_save_merge_qwen2.5vl32B_model_ocr_benchmark.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 287 |
| Functions | `format_data`, `find_lora_base_model` |
| Imports | datasets, jiwer, os, pandas, pathlib, qwen_vl_utils, sys, tests, torch, tqdm, ... +2 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive benchmark test comparing OCR performance of Qwen2.5-VL-32B model across base, LoRA adapter, and merged configurations with different quantization levels.

**Mechanism:** Uses OCRModelEvaluator to systematically benchmark: (1) base Qwen2.5-VL-32B-Instruct-bnb-4bit model establishing baseline WER/CER metrics, (2) applies LoRA (r=16, vision+language layers) and trains for 60 steps on French OCR data then evaluates adapter performance, (3) merges LoRA into base model using save_pretrained_merged creating 16-bit merged checkpoint, (4) evaluates merged model in three loading modes: 16-bit (full precision), 4-bit quantized, and 8-bit quantized, comparing WER (Word Error Rate) and CER (Character Error Rate) across all five configurations. Uses find_lora_base_model helper to extract base model class, generates comparison report showing performance deltas, and cleans up all artifacts.

**Significance:** Critical benchmark for validating that model merging preserves OCR quality in large vision models. Tests whether merging LoRA adapters degrades accuracy compared to adapter-only inference, and whether re-quantizing merged models (4-bit/8-bit) maintains acceptable performance. Essential for production deployment decisions where memory/speed tradeoffs must be balanced against accuracy requirements in vision tasks.
