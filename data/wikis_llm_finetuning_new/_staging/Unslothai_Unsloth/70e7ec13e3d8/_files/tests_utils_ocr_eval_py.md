# File: `tests/utils/ocr_eval.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 374 |
| Classes | `OCRModelEvaluator` |
| Functions | `evaluate_ocr_model`, `create_evaluator` |
| Imports | jiwer, matplotlib, os, pandas, qwen_vl_utils, torch, tqdm, traceback, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides a comprehensive OCR model evaluation framework for benchmarking vision-language models on text extraction tasks using WER and CER metrics.

**Mechanism:** The `OCRModelEvaluator` class processes datasets with message structures containing images and ground truth text, extracts user/assistant/system messages via `_extract_sample_components()`, generates model responses using Qwen-specific processing (`process_vision_info()`, `apply_chat_template()`), calculates Word Error Rate and Character Error Rate using the jiwer library, saves per-sample results to text files, aggregates metrics to CSV/summary files, and supports multi-model comparison with `add_to_comparison()`, `print_model_comparison()` and matplotlib bar chart visualizations. Convenience functions `evaluate_ocr_model()` and `create_evaluator()` provide backward-compatible API.

**Significance:** Specialized evaluation utility for testing Unsloth's vision-language model capabilities, particularly for Qwen-VL based OCR fine-tuning scenarios where text extraction accuracy is the primary metric.
