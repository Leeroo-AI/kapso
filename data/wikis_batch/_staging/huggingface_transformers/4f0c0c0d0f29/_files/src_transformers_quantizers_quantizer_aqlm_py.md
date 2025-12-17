# File: `src/transformers/quantizers/quantizer_aqlm.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 73 |
| Classes | `AqlmHfQuantizer` |
| Imports | base, importlib, integrations, packaging, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements quantization support for AQLM (Additive Quantization for Language Models), a method that uses additive quantization techniques for model compression. This quantizer enables loading and potentially training AQLM-quantized models.

**Mechanism:** The `AqlmHfQuantizer` class extends `HfQuantizer` and sets `requires_calibration = True` indicating models must be pre-quantized. During preprocessing, it calls `replace_with_aqlm_linear()` to swap standard linear layers with AQLM-specific implementations. The environment validation checks for accelerate and aqlm library dependencies. Training support is conditional on AQLM library version (>= 1.0.2), checked via version parsing.

**Significance:** Enables use of AQLM quantization method in Transformers, providing an alternative compression approach. AQLM offers competitive compression ratios and inference speed. The version-gated training support shows active development and evolving capabilities of the quantization method.
