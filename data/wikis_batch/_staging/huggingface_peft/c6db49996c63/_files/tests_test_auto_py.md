# File: `tests/test_auto.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 225 |
| Classes | `TestPeftAutoModel` |
| Imports | peft, tempfile, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for AutoPeftModel classes for automatic model loading.

**Mechanism:** Contains `TestPeftAutoModel` class testing automatic PEFT model loading for different task types: CausalLM, Seq2SeqLM, SequenceClassification, TokenClassification, QuestionAnswering, FeatureExtraction, and Whisper. Validates checkpoint loading, kwargs passing (dtype, adapter_name, is_trainable), and vocabulary handling (extended vocab, spare embeddings like Qwen models).

**Significance:** Ensures the Auto classes correctly infer and load appropriate PEFT model types from checkpoints, supporting seamless model loading workflows similar to transformers' AutoModel pattern.
