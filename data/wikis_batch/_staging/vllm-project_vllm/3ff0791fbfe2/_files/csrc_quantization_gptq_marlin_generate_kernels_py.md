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

**Purpose:** Generates CUDA kernel instantiations for GPTQ-Marlin quantization kernels supporting various data type combinations and activation precisions.

**Mechanism:** Similar to marlin_moe_wna16 generator but adds support for INT8 and FP8 activations alongside standard FP16/BF16. Uses Jinja2 to generate kernel files and a kernel_selector.h dispatch header. Includes additional HQQ quantization support with floating-point zero points.

**Significance:** Build-time code generator for GPTQ quantization kernels. The comprehensive type combination coverage (12+ configurations) enables broad quantization strategy support while maintaining kernel specialization benefits. Essential for GPTQ model deployment.
