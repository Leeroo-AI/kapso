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

**Purpose:** OCR evaluation framework

**Mechanism:** Evaluates OCR models using WER and CER metrics, generates detailed reports with visualizations and model comparisons

**Significance:** Core utility for validating OCR model quality across vision tests
