# File: `tests/test_lora_variants.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 316 |
| Classes | `CustomModel`, `DummyLM`, `MockTransformerWrapper`, `TestLoraVariants`, `TestActivatedLora` |
| Imports | peft, pytest, testing_common, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for LoRA variants including DoRA and aLoRA (Activated LoRA).

**Mechanism:** Tests DoRA variant application across different layer types (Linear, Embedding, Conv1d, Conv2d) and verifies gradient flow through DoRA parameters. Tests aLoRA offset calculation for conditional activation based on invocation tokens, ensuring adapter remains inactive before invocation point. Verifies gradient checkpointing compatibility and beam search restrictions.

**Significance:** Validates specialized LoRA variants that modify standard LoRA behavior - DoRA for improved training dynamics through magnitude/direction decomposition, and aLoRA for conditional adapter activation based on input tokens.
