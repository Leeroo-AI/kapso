# File: `src/peft/tuners/randlora/bnb.py`

**Category:** quantization

| Property | Value |
|----------|-------|
| Lines | 457 |
| Classes | `Linear8bitLt`, `Linear4bit` |
| Imports | __future__, bitsandbytes, layer, peft, torch, typing, warnings |

## Understanding

**Status:** ✅ Fully explored

**Purpose:** Quantized variants of RandLora layers supporting BitsAndBytes 8-bit and 4-bit quantization for memory-efficient inference and training.

**Mechanism:**
- **Linear8bitLt class** (8-bit quantization):
  - Inherits from `torch.nn.Module` and `RandLoraLayer`
  - Base layer is `bnb.nn.Linear8bitLt` (8-bit quantized weights)
  - **Key Methods:**
    - `merge()`: Dequantizes base weights, adds delta, requantizes
      - Uses `dequantize_bnb_weight()` to convert int8 to float
      - Warns about potential rounding errors
      - Creates new Int8Params with merged weights
    - `unmerge()`: Reverses merge operation
    - `get_scaled_bases()`: Identical to standard Linear (handles scaling and slicing)
    - `get_delta_weight()`: Computes delta weight for merging
    - `forward()`: Applies RandLora to quantized base layer
      - Handles dtype conversion for quantized computation
      - Uses autocast detection for proper dtype management
  - Supports safe_merge option to check for NaNs

- **Linear4bit class** (4-bit quantization):
  - Inherits from `torch.nn.Module` and `RandLoraLayer`
  - Base layer is `bnb.nn.Linear4bit` (4-bit quantized weights)
  - **Key Methods:**
    - `merge()`: Dequantizes with `bnb.functional.dequantize_4bit()`, adds delta, requantizes
      - Creates Params4bit with merged weights
      - Preserves quantization config (compute_dtype, compress_statistics, quant_type)
      - Warns about potential rounding errors
    - `unmerge()`: Reverses merge operation
    - `get_scaled_bases()`: Identical to standard Linear
    - `get_delta_weight()`: Computes delta weight
    - `forward()`: Applies RandLora to 4-bit quantized layer
      - Clones result defensively (required for 4-bit backprop)
      - Handles autocast and dtype conversion
  - Both merge operations include safe_merge option

- **Common Patterns:**
  - Both classes share the `get_scaled_bases()` method from standard Linear via RandLoraLayer
  - Use `UniqueBaseGrad.apply()` for memory-efficient scaling
  - Handle CPU bf16/fp16 edge cases by casting to fp32 for merge operations
  - Preserve base layer quantization state during merge/unmerge

**Significance:** Enables RandLora to work with quantized models for extreme memory efficiency. Combining RandLora's parameter reduction with quantization allows:
1. **8-bit**: ~4x memory reduction on weights
2. **4-bit**: ~8x memory reduction on weights
3. **RandLora**: ~10x reduction in trainable parameters
Together, this enables fine-tuning very large models on consumer hardware. The careful handling of quantization states ensures numerical stability during merge operations, though some rounding errors are inevitable when converting between quantized and full-precision representations.
