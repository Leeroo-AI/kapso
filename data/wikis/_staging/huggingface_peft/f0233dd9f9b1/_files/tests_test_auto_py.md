# File: `tests/test_auto.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 225 |
| Classes | `TestPeftAutoModel` |
| Imports | peft, tempfile, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for AutoPeftModel classes

**Mechanism:** Tests AutoPeftModel automatic model loading for different task types (CausalLM, Seq2SeqLM, SequenceClassification, TokenClassification, QuestionAnswering, FeatureExtraction) including save/load, kwargs passing, and extended vocabulary support

**Significance:** Test coverage for automatic PEFT model loading functionality
