# File: `csrc/quantization/gptq_marlin/generate_kernels.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 316 |
| Functions | `remove_old_kernels`, `generate_new_kernels` |
| Imports | glob, itertools, jinja2, os, subprocess, sys |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Code generator for GPTQ-Marlin quantized GEMM kernels supporting various precision formats and mixed activations.

**Mechanism:** Similar to marlin_moe but for dense (non-MoE) models. Generates kernel instantiations for: AWQ-INT4, GPTQ-INT4/INT8, FP8, NVFP4, MXFP4, HQQ, plus mixed precision configs (INT8/FP8 activations with INT4 weights). Includes kernel_selector.h for runtime dispatch based on type configurations. Supports floating-point zero points (is_zp_float) for HQQ.

**Significance:** Foundation for GPTQ/AWQ quantization support in vLLM. The generated kernels provide high-performance inference for compressed models, enabling deployment of larger models on limited GPU memory.
