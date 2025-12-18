# File: `src/transformers/quantizers/quantizer_spqr.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 79 |
| Classes | `SpQRHfQuantizer` |
| Imports | base, integrations, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements the SpQR (Sparse-Quantized Representation) quantization method, enabling loading and inference of pre-quantized models using sparse quantization techniques.

**Mechanism:** The `SpQRHfQuantizer` class extends `HfQuantizer` to support SpQR quantization. It validates that CUDA is available (GPU required), checks for both accelerate and spqr_quant[gpu] libraries, and enforces torch.float16 dtype exclusively. During model preparation, it identifies modules to exclude from quantization (combining user-specified exclusions with `_keep_in_fp32_modules`), then calls `replace_with_spqr_linear()` to replace standard linear layers with SpQR-quantized equivalents before weights are loaded. The quantizer requires pre-calibrated models and does not support on-the-fly quantization.

**Significance:** Adds SpQR quantization support to transformers, providing a sparse quantization approach that can achieve high compression ratios while maintaining model quality. SpQR is particularly effective for aggressive quantization scenarios where traditional dense quantization methods struggle. The quantizer is trainable-false but serializable, allowing saved SpQR models to be shared and reloaded efficiently.
