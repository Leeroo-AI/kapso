# File: `src/transformers/pipelines/fill_mask.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 259 |
| Classes | `FillMaskPipeline` |
| Imports | base, numpy, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Predicts masked tokens in text using masked language models like BERT, RoBERTa, or ALBERT.

**Mechanism:** FillMaskPipeline tokenizes input with mask tokens, validates exactly one mask per sequence via ensure_exactly_one_mask_token(), runs through masked LM to get logits, applies softmax to get probabilities, optionally filters to target_ids vocabulary subset, returns top-k predictions with scores, decoded tokens, and complete sequences with mask filled.

**Significance:** Core for demonstrating masked language model capabilities, enabling text completion, word prediction, and cloze test applications. Supports experimentation with model vocabulary knowledge and can be used for data augmentation, error correction, and exploring semantic alternatives.
