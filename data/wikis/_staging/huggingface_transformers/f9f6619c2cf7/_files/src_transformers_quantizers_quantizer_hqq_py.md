# File: `src/transformers/quantizers/quantizer_hqq.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 262 |
| Classes | `HqqHfQuantizer` |
| Imports | base, integrations, quantizers_utils, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements Half-Quadratic Quantization (HQQ) method, a flexible quantization approach supporting various bit-widths and configurations without requiring calibration data.

**Mechanism:** Extends HfQuantizer integrating with external hqq library (>= 0.2.1). Patches HQQLinear with dummy weight property for dtype/device compatibility. Before loading, tags Linear modules with appropriate quant_config via prepare_for_hqq_linear. Unlike most quantizers, treats all Linear layer parameters as needing quantization since modules aren't fully prepared upfront. Includes special forward patching for multi-GPU setups to handle device mismatches. Sets model flags (is_hqq_quantized, is_hqq_serializable) after loading.

**Significance:** Flexible quantizer supporting arbitrary quantization configurations per-layer or globally. Trainable and serializable with multi-GPU support. Notable for not requiring calibration data and supporting CPU/disk devices (though with warnings), making it accessible for various deployment scenarios.
