# File: `tests/test_adaption_prompt.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 416 |
| Classes | `TestAdaptionPrompt` |
| Imports | os, peft, pytest, tempfile, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for AdaptionPrompt (llama-adapter) functionality.

**Mechanism:** Contains the `TestAdaptionPrompt` class with comprehensive tests covering: model attributes, training preparation, 8-bit quantization support, checkpoint saving/loading, generation, adapter operations (add/set/disable), use_cache behavior, bf16 inference, and multi-adapter workflows across multiple model architectures (GPT2, Llama, Mistral).

**Significance:** Validates the AdaptionPrompt implementation for efficient prompt tuning on causal language models, ensuring proper integration with quantization methods and correct adapter state management.
