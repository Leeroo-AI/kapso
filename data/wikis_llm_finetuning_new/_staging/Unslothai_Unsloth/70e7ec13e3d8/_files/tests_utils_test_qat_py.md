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

**Purpose:** Unit tests verifying Quantization-Aware Training (QAT) fake quantizer integration for both full fine-tuning and LoRA models using fp8-int4 and fp8-fp8 schemes.

**Mechanism:** Defines `_CountingFakeQuantizer` to track quantizer invocations. `_get_model()` loads Qwen3-1.7B via FastLanguageModel with specified qat_scheme, returning PEFT model if not full_finetuning. `_test_linear_is_fake_quantized()` verifies FakeQuantizedLinear wrappers with correct quantizer classes (Float8FakeQuantizer for activations, Int4WeightPreshuffledFakeQuantizer or Float8FakeQuantizer for weights depending on scheme), checking both base layers and LoRA A/B modules with minimum in_features threshold (128 for fp8-int4). `_test_fake_quantizers_are_called()` swaps quantizers for _CountingFakeQuantizer instances, runs forward pass, then asserts each was called exactly once, with special handling for LoRA's shared activation quantization per attention/MLP block. Parameterized pytest tests validate both qat_scheme variants.

**Significance:** Essential tests for Unsloth's QAT feature enabling quantization-aware fine-tuning, verifying correct torchao FakeQuantizedLinear integration for training models that will be deployed with int4/fp8 quantization.
