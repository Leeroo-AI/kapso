# File: `src/peft/tuners/road/__init__.py`

**Category:** initialization

| Property | Value |
|----------|-------|
| Lines | 48 |
| Imports | config, layer, model, peft |

## Understanding

**Status:** ✅ Fully explored

**Purpose:** Package initialization file that exposes the RoAd (Rotation Adapter) PEFT method and provides lazy loading for quantized variants.

**Mechanism:**
- Imports and re-exports core RoAd components: `RoadConfig`, `RoadLayer`, `RoadModel`, and `Linear`
- Registers RoAd with PEFT using `register_peft_method()` with `is_mixed_compatible=True` (supports mixed adapter batches)
- Implements `__getattr__()` for lazy loading of quantized variants:
  - `Linear8bitLt`: 8-bit quantized variant
  - `Linear4bit`: 4-bit quantized variant
- Defines `__all__` for explicit export control
- Includes attribution comment noting implementation is based on non-author community work

**Significance:** Essential initialization that makes RoAd available as a PEFT method with quantization support and mixed-batch inference. RoAd is a parameter-efficient adapter that uses 2D rotation matrices with trainable angles and scales to transform hidden states. The method achieves strong performance with minimal parameters by applying learned rotations to pairs of elements in the activation vectors. Paper reference: https://huggingface.co/papers/2409.00119.
