# File: `src/transformers/quantizers/quantizer_fbgemm_fp8.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 187 |
| Classes | `FbgemmFp8HfQuantizer` |
| Imports | base, quantizers_utils, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements FP8 quantization using FBGEMM kernels for high-performance inference on modern GPUs with compute capability >= 9.0 (H100 and newer).

**Mechanism:** Extends HfQuantizer to replace Linear layers with FbgemmFp8Linear or FbgemmFp8Llama4TextExperts modules. Enforces bfloat16 dtype and validates GPU compute capability. Implements sophisticated tensor parallelism plans for Llama4 models with special handling for attention layers (local_colwise/local_rowwise with gather operations) and MoE experts (including fused implementations). Supports both CUDA (via fbgemm-gpu) and XPU (via kernels) backends.

**Significance:** High-performance quantizer specifically for cutting-edge GPUs, targeting production inference at scale. Non-trainable but serializable. Extensive TP support makes it suitable for distributed inference of large models, particularly Llama4 variants.
