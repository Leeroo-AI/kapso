# File: `src/peft/tuners/lora/torchao.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 156 |
| Classes | `TorchaoLoraLinear` |
| Functions | `dispatch_torchao` |
| Imports | __future__, config, layer, peft, torch, typing, warnings |

## Understanding

**Status:** ✅ Documented

**Purpose:** LoRA adapter for TorchAO (PyTorch Architecture Optimization) quantized tensors

**Mechanism:** TorchaoLoraLinear handles TorchAO's AffineQuantizedTensor and LinearActivationQuantizedTensor formats (int8 weight-only quantization). Implements merge/unmerge by dequantizing via TorchAO's dequantize() method, applying LoRA deltas, then requantizing using TorchAO's quantize_ function with the original quantization configuration.

**Significance:** Integrates LoRA with PyTorch's native quantization optimization library TorchAO. Provides first-party PyTorch support for quantized LoRA training, important for users preferring PyTorch's official quantization path over third-party libraries, and enables LoRA on models using TorchAO's optimized inference kernels.
