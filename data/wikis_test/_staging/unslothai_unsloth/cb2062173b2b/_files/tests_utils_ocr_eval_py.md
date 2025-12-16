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

**Purpose:** Provides comprehensive evaluation framework for OCR (Optical Character Recognition) models using Word Error Rate (WER) and Character Error Rate (CER) metrics. Specifically designed for vision-language models that can extract text from images.

**Mechanism:** The `OCRModelEvaluator` class orchestrates the evaluation process:
- Processes datasets with multi-modal messages containing images, questions, and ground truth text
- Extracts components from messages (system, user with image/question, assistant with ground truth)
- Generates model responses using Qwen vision utilities (`process_vision_info`, `apply_chat_template`)
- Calculates WER and CER metrics using the `jiwer` library for each sample
- Saves detailed results including per-sample outputs and aggregate metrics to files
- Supports model comparison tracking with visualization (bar charts) and CSV export
- Includes error handling with detailed traceback for debugging failed samples

The convenience function `evaluate_ocr_model` provides backward compatibility with the original API.

**Significance:** Specialized evaluation utility for testing vision-language models on OCR tasks. This is important for validating unsloth's support for multi-modal models like Qwen-VL, ensuring they can accurately extract text from images. The comparison features enable benchmarking multiple models side-by-side.
