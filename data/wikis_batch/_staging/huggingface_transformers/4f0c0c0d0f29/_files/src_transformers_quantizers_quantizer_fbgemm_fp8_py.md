# File: `src/transformers/quantizers/quantizer_fbgemm_fp8.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 187 |
| Classes | `FbgemmFp8HfQuantizer` |
| Imports | base, quantizers_utils, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements FP8 (8-bit floating point) quantization using Facebook's FBGEMM library, targeting high-end GPUs (compute capability >= 9.0, e.g., H100) and Intel XPU accelerators. Provides efficient FP8 inference with specialized kernel support.

**Mechanism:** The `FbgemmFp8HfQuantizer` requires CUDA >=9.0 or XPU with appropriate libraries (fbgemm-gpu for CUDA, kernels for XPU). Forces bfloat16 dtype for compatibility. During preprocessing, `replace_with_fbgemm_fp8_linear()` converts layers to `FbgemmFp8Linear` or `FbgemmFp8Llama4TextExperts` for MoE models. Extensive tensor parallel plan configuration for Llama4, defining local_colwise/local_rowwise strategies for attention, feed-forward, and expert layers with corresponding scale tensors. Not trainable but serializable. Provides `FbgemmFp8Quantize` operations for on-the-fly quantization.

**Significance:** FP8 represents the next generation of quantization for cutting-edge hardware, offering near-FP16 accuracy with INT8 memory footprint. The H100-specific optimization reflects deployment targets for large-scale production systems. Complex tensor parallelism integration shows sophisticated distributed inference support, particularly important for large MoE models.
