# File: `src/peft/tuners/vera/layer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 291 |
| Classes | `VeraLayer`, `Linear` |
| Imports | _buffer_dict, peft, torch, transformers, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements VeRA adapter layers with shared projection matrices and per-layer trainable scaling vectors.

**Mechanism:** VeraLayer maintains shared vera_A/vera_B buffers (references to model-wide projections) and per-adapter trainable parameters vera_lambda_b (out_features) and vera_lambda_d (rank). Linear class performs forward pass via: output = base_layer(x) + lambda_b * F.linear(lambda_d * F.linear(x, sliced_A), sliced_B). Slicing allows reusing oversized projections across different layer dimensions. Delta weight computed as (lambda_b * sliced_B) @ (lambda_d * sliced_A).

**Significance:** Core implementation of VeRA's parameter efficiency - only 2*r + out_features trainable parameters per layer versus LoRA's r*(in_features + out_features). Shared projections drastically reduce storage when adapting multiple layers while maintaining competitive performance.
