# File: `src/transformers/quantizers/quantizer_mxfp4.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 292 |
| Classes | `Mxfp4HfQuantizer` |
| Imports | base, quantizers_utils, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements MXFP4 (Microscaling FP4) quantization using block-wise floating-point formats with dynamic exponents, targeting GPUs >= compute capability 7.5 or Intel XPUs.

**Mechanism:** Extends HfQuantizer with lazy kernel loading from kernels-community/triton_kernels hub. Validates compute capability (CUDA >= 7.5, requires Triton >= 3.4.0; XPU requires Triton >= 3.5.0) and gracefully falls back to bf16 dequantization on unsupported hardware. Replaces modules with Mxfp4GptOssExperts for MoE models. Implements sophisticated weight conversion with swizzled/unswizzled data layouts for _blocks and _scales parameters. Custom TP/EP plans use grouped_gemm for expert parallelism. Overrides get_state_dict_and_metadata to handle storage layout transformations during serialization.

**Significance:** Modern quantizer for block-wise microscaling FP4, balancing accuracy and compression. Non-trainable but serializable. Broader GPU support than cutting-edge quantizers (T4/A100/L4/H100) makes it more accessible. Sophisticated handling of weight layouts and expert parallelism shows optimization for MoE architectures.
