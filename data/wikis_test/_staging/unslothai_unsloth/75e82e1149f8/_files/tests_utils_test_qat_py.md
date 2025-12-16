# File: `tests/utils/test_qat.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 156 |
| Classes | `_CountingFakeQuantizer` |
| Functions | `test_lora_model_fake_quantize` |
| Imports | pytest, torch, torchao, typing, unsloth |

## Understanding

**Status:** ✅ Explored

**Purpose:** Quantization-aware training tests

**Mechanism:** Tests QAT schemes (fp8-int4, fp8-fp8) for both full finetuning and LoRA, verifies fake quantizers are applied correctly

**Significance:** Validates quantization-aware training functionality
