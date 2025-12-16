# File: `tests/utils/test_qat.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 156 |
| Classes | `_CountingFakeQuantizer` |
| Functions | `test_lora_model_fake_quantize` |
| Imports | pytest, torch, torchao, typing, unsloth |

## Understanding

**Status:** ✅ Documented

**Purpose:** Tests quantization-aware training infrastructure

**Mechanism:** Validates FakeQuantizedLinear integration with LoRA models for QAT

**Significance:** Ensures QAT infrastructure works correctly for training quantization-aware models
