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

**Purpose:** OCR model evaluation framework providing comprehensive assessment of vision language models on OCR tasks using Word Error Rate (WER) and Character Error Rate (CER) metrics.

**Mechanism:** OCRModelEvaluator class orchestrates evaluation pipeline: extracts components from message-formatted samples (system/user/assistant messages with images), generates model responses using Qwen's vision processing pipeline (process_vision_info for images, apply_chat_template for prompts), calculates WER/CER using jiwer library, saves individual results and detailed CSV reports, tracks multiple model comparisons, and generates matplotlib visualizations comparing model performance. Handles errors gracefully with detailed tracebacks.

**Significance:** Standardizes OCR evaluation across vision model tests, ensuring consistent metrics and reporting. Essential for the vision model benchmark tests that compare base models, LoRA adapters, and merged models across different quantization levels. The comparison functionality enables systematic analysis of how different training and merging approaches affect OCR quality. Critical for validating that Unsloth's optimizations don't degrade vision model performance.
