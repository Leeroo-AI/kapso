# File: `unsloth/kernels/layernorm.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 225 |
| Classes | `Fast_Layernorm` |
| Functions | `layernorm_forward`, `layernorm_backward`, `fast_layernorm`, `test_layernorm`, `testing_suite_layernorm` |
| Imports | torch, triton, unsloth_zoo, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Accelerates standard LayerNorm computation with fused forward/backward Triton kernels using float32 internal precision for numerical stability.

**Mechanism:** Triton kernels compute mean and variance in a single pass (1/N * sum(X), 1/N * sum((X-mean)^2)) and apply: output = (X-mean)*rsqrt(var+eps)*W + b. Backward pass derives dX using chain rule accounting for mean/variance computation. All internal computation in float32 regardless of input dtype (as per PyTorch TorchTune). Grid dimension matches number of rows for parallel processing.

**Significance:** Provides ~1.5-2x speedup for LayerNorm through kernel fusion, reducing latency bottleneck that becomes significant in inference with small batch sizes.
