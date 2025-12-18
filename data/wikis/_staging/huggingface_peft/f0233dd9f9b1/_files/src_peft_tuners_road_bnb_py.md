# File: `src/peft/tuners/road/bnb.py`

**Category:** quantization

| Property | Value |
|----------|-------|
| Lines | 408 |
| Classes | `Linear8bitLt`, `Linear4bit` |
| Functions | `dispatch_bnb_8bit`, `dispatch_bnb_4bit` |
| Imports | __future__, bitsandbytes, config, layer, peft, torch, typing, warnings |

## Understanding

**Status:** ✅ Fully explored

**Purpose:** Quantized variants of RoAd layers supporting BitsAndBytes 8-bit and 4-bit quantization for memory-efficient inference and training.

**Mechanism:**
- **Linear8bitLt class** (8-bit quantization):
  - Inherits from `torch.nn.Module` and `RoadLayer`
  - Base layer is `bnb.nn.Linear8bitLt`
  - **merge()**: Dequantizes weights, applies R @ W transformation, requantizes
    - Uses `dequantize_bnb_weight()` for int8 -> float conversion
    - Applies rotation to both weights and bias
    - Warns about potential rounding errors from quantization cycle
  - **unmerge()**: Applies R^-1 @ W to reverse merge
    - Computes inverse via `torch.linalg.inv()`
  - **forward()**: Applies rotation to output of quantized layer
    - Handles dtype conversion carefully for quantized computation
  - **dispatch_bnb_8bit()**: Factory function with state preservation

- **Linear4bit class** (4-bit quantization):
  - Inherits from `torch.nn.Module` and `RoadLayer`
  - Base layer is `bnb.nn.Linear4bit`
  - **merge()**: Dequantizes with `bnb.functional.dequantize_4bit()`, applies R @ W, requantizes
    - Preserves quantization config (compute_dtype, compress_statistics, quant_type)
    - Cleans kwargs (removes bnb_quantized flag, _* attributes)
    - Applies rotation to weights and bias
  - **unmerge()**: Uses R^-1 to reverse merge
  - **forward()**: Applies rotation to 4-bit quantized layer output
    - Note: Commented out defensive clone (not needed in current PyTorch)
  - **dispatch_bnb_4bit()**: Factory function checking is_bnb_4bit_available()

- **Common Patterns:**
  - Both classes reuse _apply_road and _get_delta_weight from base layer module
  - Merge operations include safe_merge option to check for NaNs
  - Handle CPU bf16/fp16 edge cases by casting to fp32
  - Inverse computation uses float32 for numerical stability

**Significance:** Enables RoAd to work with quantized models for extreme memory efficiency:
1. **8-bit**: ~4x memory reduction on weights
2. **4-bit**: ~8x memory reduction on weights
3. **RoAd**: Minimal adapter parameters (e.g., hidden_dim/2 for road_1)
Together, this enables adapting very large models on consumer hardware. The rotation matrix merging (R @ W) is particularly elegant for quantized models as it maintains the linear transformation property. The careful handling of quantization states and inverse computation ensures numerical stability. Note: merge/unmerge involves quantization cycles that may introduce small rounding errors.
