# File: `src/transformers/quantizers/quantizer_finegrained_fp8.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 162 |
| Classes | `FineGrainedFP8HfQuantizer` |
| Imports | base, quantizers_utils, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements fine-grained FP8 quantization supporting both standard models and MoE architectures, with e4m3fn format support. Provides FP8 quantization accessible on broader GPU hardware (compute capability >= 8.9) compared to FBGEMM, including consumer GPUs like RTX 4090.

**Mechanism:** The `FineGrainedFP8HfQuantizer` supports CUDA (>=8.9) or XPU platforms. Includes automatic dequantization fallback to bfloat16 when no GPU/XPU is available for pre-quantized models. During preprocessing, `replace_with_fp8_linear()` converts layers to `FP8Linear` or `FP8Expert` modules. Implements tensor parallel plans for Qwen3 models with local_colwise/local_rowwise strategies and scale inverse tensors. Provides `Fp8Quantize` operations and `Fp8Dequantize` weight conversion when dequantize flag is set. Not trainable but serializable. Returns warmup factor of 2 indicating clean preprocessing.

**Significance:** Democratizes FP8 quantization by supporting wider range of GPUs including consumer hardware (RTX 4090), making advanced FP8 compression accessible beyond H100 data center GPUs. The automatic dequantization fallback enables flexible deployment. Fine-grained quantization provides better accuracy than coarser approaches at similar compression ratios.
