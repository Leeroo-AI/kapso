# File: `src/peft/tuners/lora/inc.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 78 |
| Classes | `IncLoraLinear` |
| Functions | `dispatch_inc` |
| Imports | layer, peft, torch, typing |

## Understanding

**Status:** ✅ Documented

**Purpose:** LoRA adapter for Intel Neural Compressor (INC) quantized models

**Mechanism:** IncLoraLinear handles Intel's Neural Compressor quantization format, supporting Intel-optimized inference backends. Detects WeightOnlyLinear layers from INC and wraps them with LoRA capability while maintaining compatibility with Intel's optimized kernels.

**Significance:** Provides LoRA support for Intel hardware acceleration and optimization paths. Important for deploying LoRA-adapted models on Intel architectures (Xeon, Gaudi) where INC quantization delivers optimal performance, enabling efficient fine-tuning in Intel-centric deployment environments.
