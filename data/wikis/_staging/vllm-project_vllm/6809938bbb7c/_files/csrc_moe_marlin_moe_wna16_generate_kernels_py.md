# File: `csrc/moe/marlin_moe_wna16/generate_kernels.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 306 |
| Functions | `remove_old_kernels`, `generate_new_kernels` |
| Imports | glob, itertools, jinja2, os, subprocess, sys |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Code generator for Marlin MoE (Mixture of Experts) quantized GEMM kernels with WnA16 (weight n-bit, activation 16-bit) support.

**Mechanism:** Uses Jinja2 templates to generate CUDA kernel instantiations for different configurations: quantization types (AWQ-INT4, GPTQ-INT4, AWQ-INT8, FP8, NVFP4, MXFP4), thread configs (128-256 threads), M-block sizes, and group sizes. Generates separate .cu files for SM75/SM80/SM89 architectures with appropriate stage counts and FP8 support detection.

**Significance:** Enables efficient MoE inference on quantized models by generating optimized kernels for all supported hardware/quantization combinations. Reduces code duplication and ensures consistent coverage across the configuration space.
