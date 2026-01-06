# File: `src/transformers/quantizers/quantizer_bitnet.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 109 |
| Classes | `BitNetHfQuantizer` |
| Imports | base, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements 1.58-bit quantizer for BitNet, an extreme quantization method using ternary weights (-1, 0, +1).

**Mechanism:** The `BitNetHfQuantizer` class extends `HfQuantizer` with `requires_calibration=True`. `validate_environment` checks for accelerate and warns if no GPU available (CPU inference slow due to unpacking overhead). It validates device_map to prevent CPU/disk offloading which isn't supported. In preprocessing, `replace_with_bitnet_linear` converts linear layers to BitLinear layers. `adjust_max_memory` reduces available memory to 90% for quantization buffers. `adjust_target_dtype` sets torch.int8 for memory calculations. Training and QAT supported only with autobitlinear class in online mode.

**Significance:** Enables BitNet's ultra-aggressive 1.58-bit quantization for maximum compression. Represents the extreme end of quantization spectrum, useful for deployment in severely memory-constrained environments while maintaining reasonable model quality through ternary weight representation.
