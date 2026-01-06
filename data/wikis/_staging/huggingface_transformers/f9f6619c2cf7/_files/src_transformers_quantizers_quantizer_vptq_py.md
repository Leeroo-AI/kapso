# File: `src/transformers/quantizers/quantizer_vptq.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 73 |
| Classes | `VptqHfQuantizer` |
| Imports | base, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements VPTQ (Vector Post-Training Quantization) method support, enabling loading and inference of pre-quantized models using the VPTQ quantization technique.

**Mechanism:** The `VptqHfQuantizer` class extends `HfQuantizer` to support VPTQ quantization. It validates that CUDA is available (GPU required), checks for both accelerate and vptq (≥0.0.4) libraries during environment validation. Before weight loading, it identifies modules to exclude from quantization by combining user-specified patterns, config-specified modules, and the model's `_keep_in_fp32_modules` list, then calls `replace_with_vptq_linear()` from integrations to replace standard linear layers with VPTQ-quantized equivalents. Like SpQR, it requires calibration and does not support on-the-fly quantization.

**Significance:** Adds VPTQ quantization method to transformers' quantization toolkit, providing another advanced post-training quantization approach for extreme compression scenarios. VPTQ uses vector-based quantization techniques that can achieve efficient compression while maintaining model quality. The quantizer is not trainable but is serializable, allowing pre-quantized VPTQ models to be saved and redistributed efficiently through the transformers ecosystem.
