# File: `src/peft/tuners/lora/inc.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 78 |
| Classes | `IncLoraLinear` |
| Functions | `dispatch_inc` |
| Imports | layer, peft, torch, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Intel Neural Compressor LoRA

**Mechanism:** Implements LoRA for Intel Neural Compressor (INC) FP8-quantized models. IncLoraLinear inherits from standard Linear LoRA layer and wraps INC PatchedLinear layers. Uses default LoRA forward pass behavior from parent class. Explicitly raises NotImplementedError for merge/unmerge operations as INC integration is still under development. dispatch_inc() detects PatchedLinear from neural_compressor and creates wrappers. Tests are maintained in Optimum-Habana repository.

**Significance:** Provides initial LoRA support for Intel's Neural Compressor framework, which optimizes models for Intel hardware with FP8 quantization. Important for users deploying on Intel accelerators (Habana Gaudi). Currently experimental with limited functionality, but enables basic fine-tuning on INC-optimized models.
