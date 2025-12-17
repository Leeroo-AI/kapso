# File: `tests/test_common_gpu.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 2185 |
| Classes | `PeftGPUCommonTests`, `TestSameAdapterDifferentDevices`, `MLP`, `ModelEmbConv1D`, `ModelConv2D` |
| Imports | accelerate, gc, parameterized, peft, pytest, tempfile, testing_utils, torch, transformers, unittest |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for GPU-specific operations across PEFT methods.

**Mechanism:** Contains two main test classes: `PeftGPUCommonTests` validates 4-bit/8-bit quantization with multiple PEFT methods (LoRA, VeRA, RandLora, IA3, Road), multi-GPU inference, adapter merging/unmerging, DoRA with quantization, ephemeral GPU offload, HRA inference, and mixed adapter batches; `TestSameAdapterDifferentDevices` ensures adapters on different devices don't interfere when adding new adapters (issue #1639).

**Significance:** Critical for validating PEFT's integration with quantization libraries (bitsandbytes, GPTQ), ensuring correct behavior in multi-GPU setups, and verifying memory-efficient inference patterns like ephemeral GPU offload for DoRA.
