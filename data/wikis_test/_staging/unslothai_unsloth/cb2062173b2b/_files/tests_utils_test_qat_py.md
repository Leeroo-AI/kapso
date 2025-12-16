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

**Purpose:** Tests Quantization-Aware Training (QAT) integration with unsloth models, ensuring that fake quantizers are properly inserted into model layers for both full fine-tuning and LoRA scenarios. QAT simulates quantization during training to improve final quantized model accuracy.

**Mechanism:** The tests verify QAT integration through several helper functions:
- `_get_model()` loads models with different QAT schemes (fp8-int4 or fp8-fp8) for full fine-tuning or LoRA
- `_test_linear_is_fake_quantized()` verifies that linear layers are wrapped with `FakeQuantizedLinear` and contain the correct fake quantizers (Float8FakeQuantizer for activations, Int4WeightPreshuffledFakeQuantizer or Float8FakeQuantizer for weights)
- `_test_fake_quantizers_are_called()` uses a custom `_CountingFakeQuantizer` to verify fake quantizers are actually invoked during forward passes
- For LoRA, tests check that base layers, lora_A, and lora_B are all properly fake quantized
- The test also validates LoRA-specific behavior where input activations are only fake quantized once per attention/MLP block (at q_proj/gate_proj)

The parameterized test `test_lora_model_fake_quantize` runs against both fp8-int4 and fp8-fp8 schemes.

**Significance:** Critical for ensuring unsloth's QAT support works correctly with torchao's quantization infrastructure. QAT is important for training models that will be deployed in quantized form, as it helps models adapt to quantization during training rather than suffering accuracy loss from post-training quantization. The LoRA tests ensure efficient fine-tuning remains compatible with QAT.
