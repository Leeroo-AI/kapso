# File: `tests/test_lora_variants.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 524+ |
| Classes | `CustomModel`, `DummyLM`, `MockTransformerWrapper` |
| Imports | peft, pytest, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for LoRA variant implementations

**Mechanism:** Tests DoRA (weight-decomposed LoRA) and aLoRA (activated LoRA) across Linear, Conv1D, Conv2D, and Embedding layers, including forward pass correctness, offset calculation for generation, and variant-specific behavior

**Significance:** Test coverage for DoRA and aLoRA LoRA variants
