# File: `src/transformers/quantizers/quantizer_quark.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 115 |
| Classes | `QuarkHfQuantizer` |
| Imports | base, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements quantization support for AMD's Quark quantization library, enabling loading and inference of Quark-quantized models in transformers.

**Mechanism:** The `QuarkHfQuantizer` class extends `HfQuantizer` to integrate AMD's Quark quantization framework. It validates the Quark library is available, then uses `_map_to_quark()` to map model layers to their quantized equivalents before weight loading. The quantizer handles complex parameter naming through custom weight converters, as Quark's `QParamsLinear` modules contain separate quantizers for weights, inputs, and biases that require special state_dict key mapping (e.g., `weight_scale` maps to `weight_quantizer.scale`). The class defines `CHECKPOINT_KEYS` to handle these transformations and uses `QuarkDeserialize` operations to properly load quantized parameters while satisfying PyTorch's missing_keys validation.

**Significance:** Provides AMD Quark quantization support in transformers, enabling users to load models quantized with AMD's toolchain for inference on AMD hardware. This is a calibration-based quantizer (not on-the-fly) and is not trainable or serializable, focusing on inference with pre-quantized models. Expands the quantization ecosystem beyond GPU-focused solutions to include AMD-optimized quantization methods.
