# File: `src/transformers/quantizers/quantizer_finegrained_fp8.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 162 |
| Classes | `FineGrainedFP8HfQuantizer` |
| Imports | base, quantizers_utils, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements fine-grained FP8 quantization supporting both standard models and MoE architectures with e4m3fn format, targeting GPUs with compute capability >= 8.9.

**Mechanism:** Extends HfQuantizer to replace Linear layers with FP8Linear and FP8Expert modules. Validates compute capability (>= 8.9 for RTX 4090/H100) and supports graceful fallback to bf16 dequantization on CPU/unsupported hardware. Implements tensor parallelism with local colwise/rowwise operations and gather hooks for models like Qwen3. Uses weight_scale_inv for efficient scaling during inference. Provides weight conversion for dequantization path.

**Significance:** Versatile FP8 quantizer with broader GPU support than fbgemm variant (8.9+ vs 9.0+), making it accessible on RTX 4090 cards. Non-trainable but serializable, with special accelerator warm-up factor for efficient memory allocation. Supports both quantized inference and dequantization modes.
