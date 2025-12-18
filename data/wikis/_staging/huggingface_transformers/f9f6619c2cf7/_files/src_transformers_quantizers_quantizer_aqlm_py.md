# File: `src/transformers/quantizers/quantizer_aqlm.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 73 |
| Classes | `AqlmHfQuantizer` |
| Imports | base, importlib, integrations, packaging, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements quantizer for AQLM (Additive Quantization of Language Models) method, enabling loading of pre-quantized AQLM models.

**Mechanism:** The `AqlmHfQuantizer` class extends `HfQuantizer` with `requires_calibration=True` indicating it only supports pre-quantized models. In `validate_environment`, it checks for accelerate and aqlm library availability. The key preprocessing step in `_process_model_before_weight_loading` calls `replace_with_aqlm_linear` to replace standard linear layers with AQLM-specific quantized linear layers. The `is_trainable` property checks AQLM version (>=1.0.2 required for training support).

**Significance:** Enables integration of AQLM quantization method which uses additive quantization for extreme compression. Part of the multi-backend quantization system, allowing users to load and run models quantized with AQLM's specific approach for memory-efficient inference.
