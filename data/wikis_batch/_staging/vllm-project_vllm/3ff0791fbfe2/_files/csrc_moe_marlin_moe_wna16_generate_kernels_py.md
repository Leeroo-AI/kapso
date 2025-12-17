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

**Purpose:** Generates CUDA kernel instantiations for Marlin MoE (Mixture of Experts) quantization with 16-bit activations across multiple GPU architectures.

**Mechanism:** Uses Jinja2 templates to generate .cu files containing explicit template instantiations of Marlin kernels. Supports multiple quantization schemes (AWQ-INT4, GPTQ-INT4, AWQ-INT8, FP8, NVFP4, MXFP4) with various thread configurations and group sizes. Architecture detection from command-line args determines FP8 support (SM89/SM120) and SM75/SM80 capabilities.

**Significance:** Build-time code generator that produces optimized kernels for different hardware and quantization configurations. Enables compile-time specialization rather than runtime dispatch, improving performance for MoE models with quantized weights.
