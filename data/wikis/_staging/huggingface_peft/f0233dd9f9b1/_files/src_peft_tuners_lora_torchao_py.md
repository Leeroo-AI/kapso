# File: `src/peft/tuners/lora/torchao.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 156 |
| Classes | `TorchaoLoraLinear` |
| Functions | `dispatch_torchao` |
| Imports | __future__, config, layer, peft, torch, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** TorchAO quantized LoRA support

**Mechanism:** Implements LoRA for torchao-quantized models (PyTorch's quantization library). TorchaoLoraLinear inherits from standard Linear and wraps AffineQuantizedTensor or LinearActivationQuantizedTensor weights. Currently supports int8_weight_only quantization with dtype checking. Implements merge/unmerge by dequantizing weights, applying LoRA delta, and requantizing using quantize_() with stored tensor subclass. Does not support lora_bias. dispatch_torchao() detects quantized tensor types and creates wrappers.

**Significance:** Integrates LoRA with PyTorch's native quantization library torchao. As PyTorch's official quantization solution, torchao support is important for users staying within the PyTorch ecosystem. Provides merge/unmerge via requantization, though currently limited to int8 weights. Critical for PyTorch-first workflows.
