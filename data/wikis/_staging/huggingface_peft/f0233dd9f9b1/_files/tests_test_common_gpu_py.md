# File: `tests/test_common_gpu.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 2185 |
| Classes | `PeftGPUCommonTests`, `TestSameAdapterDifferentDevices`, `MLP`, `ModelEmbConv1D`, `ModelConv2D` |
| Imports | accelerate, gc, parameterized, peft, pytest, tempfile, testing_utils, torch, transformers, unittest |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for GPU-specific PEFT operations

**Mechanism:** Tests PEFT models on GPU including generation, 8-bit/4-bit quantization (bitsandbytes), training, multi-GPU operations, FSDP, device placement, and quantization integration with various adapters (LoRA, AdaLora, IA3, OFT, VeRA, etc.)

**Significance:** Test coverage for GPU and quantization features across adapters
