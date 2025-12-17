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

**Purpose:** Provides comprehensive OCR evaluation framework for vision-language models, measuring text extraction accuracy using standard metrics like character error rate and word error rate.

**Mechanism:** Implements OCRModelEvaluator class that runs models on image datasets, extracts predicted text, compares against ground truth using jiwer metrics (CER/WER), generates visualizations with matplotlib, creates detailed performance reports with pandas, and handles errors gracefully with traceback logging.

**Significance:** Essential for quantitatively validating vision-language model quality on document understanding tasks, providing standardized OCR benchmarking that enables objective comparison of model performance before and after fine-tuning.
