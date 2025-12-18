# File: `src/peft/tuners/road/layer.py`

**Category:** layer

| Property | Value |
|----------|-------|
| Lines | 419 |
| Classes | `RoadLayer`, `Linear` |
| Functions | `_get_delta_weight`, `_prepare_cols`, `_apply_road`, `dispatch_default` |
| Imports | torch, typing, warnings |

## Understanding

**Status:** ✅ Fully explored

**Purpose:** Implements RoAd adapter layers that transform activations using learned 2D rotations applied to element pairs.

**Mechanism:**
- **RoadLayer class** (base layer):
  - **Trainable Parameters:**
    - `road_theta`: Rotation angles for each pair/element (size depends on variant)
    - `road_alpha`: Scaling factors for each pair/element
  - `adapter_layer_names = ("road_theta", "road_alpha")`
  - `other_param_names = ("variant", "group_size")`
  - `update_layer()`: Validates group_size divides out_features, initializes theta/alpha
  - `reset_parameters()`:
    - `init_weights=True`: zeros for theta (identity), ones for alpha (no scaling)
    - `init_weights=False`: random normal for testing

- **Rotation Mathematics** (see _apply_road):
  - Splits vector into groups of size group_size
  - Within each group, pairs element i with element i+group_size/2
  - Applies 2D rotation: `[x1, x2] -> [x1*α*cos(θ) - x2*α*sin(θ), x1*α*sin(θ) + x2*α*cos(θ)]`
  - Formula: `result = x * first_col + rotate_half(x) * second_col`

- **Helper Functions:**
  - `_prepare_cols()`: Computes first_col (α*cos(θ)) and second_col (α*sin(θ))
    - road_1: Reuses parameters across pairs in group
    - road_2: One parameter set per element
    - road_4: Separate parameters for first and second columns
  - `_apply_road()`: Efficient element-wise rotation without materializing full matrix
  - `_get_delta_weight()`: Materializes full rotation matrix for merging (R @ W formula)

- **Linear class** (concrete implementation):
  - **forward()**: Three modes:
    1. `disable_adapters`: Run base layer only (unmerge if needed)
    2. `merged`: Base layer already contains merged adapter
    3. Normal: Apply RoAd rotation after base layer
    - Supports mixed-batch inference via adapter_names kwarg
  - **_mixed_batch_forward()**: Applies different adapters to different samples in batch
    - Groups samples by adapter name
    - Applies each adapter to its sub-batch
    - "__base__" special name skips adaptation
  - **merge()**: Computes R matrix, applies `W_new = R @ W` and `b_new = R @ b`
  - **unmerge()**: Computes R^-1, applies `W_original = R^-1 @ W_merged`

- **dispatch_default()**: Factory function creating Linear from torch.nn.Linear

**Significance:** Core implementation of RoAd's elegant rotation-based adaptation. Key innovations:
1. **2D rotations**: Only 2 parameters (θ, α) per pair vs full matrix
2. **Element-wise computation**: Avoids materializing rotation matrix during forward pass
3. **Grouping**: Balances flexibility with computational efficiency
4. **Merging**: Can fold into weights for zero-overhead inference via R @ W formula
5. **Mixed batches**: Different samples can use different adapters in same batch
The design achieves strong adaptation with minimal parameters: e.g., road_1 stores only hidden_dim/2 parameters per layer. The rotation-based approach is inspired by RoPE embeddings but applied as a general adaptation mechanism.
