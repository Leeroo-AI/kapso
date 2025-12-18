# File: `tests/test_adaption_prompt.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 416 |
| Classes | `TestAdaptionPrompt` |
| Imports | os, peft, pytest, tempfile, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for AdaptionPrompt adapter

**Mechanism:** Tests AdaptionPrompt functionality including adapter attributes, training preparation, int8 training support, and save/load operations across GPT2, Llama, and Mistral models

**Significance:** Test coverage for AdaptionPrompt PEFT method
