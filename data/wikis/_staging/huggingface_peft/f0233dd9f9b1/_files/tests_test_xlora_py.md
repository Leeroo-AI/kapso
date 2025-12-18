# File: `tests/test_xlora.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 473 |
| Classes | `TestXlora` |
| Functions | `flaky` |
| Imports | functools, huggingface_hub, os, peft, pytest, safetensors, testing_utils, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for X-LoRA adapter

**Mechanism:** Tests X-LoRA mixture-of-LoRAs with learned gating/routing, including multiple LoRA adapters, scalings computation, embedding support, and xlora_depth configurations

**Significance:** Test coverage for X-LoRA mixture-of-experts LoRA routing
