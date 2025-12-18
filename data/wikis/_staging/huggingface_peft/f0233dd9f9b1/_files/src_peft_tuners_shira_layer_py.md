# File: `src/peft/tuners/shira/layer.py`

**Category:** layer

| Property | Value |
|----------|-------|
| Lines | 218 |
| Classes | `ShiraLayer`, `Linear` |
| Imports | copy, peft, torch, typing, warnings |

## Understanding

**Status:** ✅ Fully explored

**Purpose:** Implements SHiRA adapter layers that apply sparse high-rank adaptation by updating only masked weight elements.

**Mechanism:**
- **ShiraLayer class** (base layer):
  - **Trainable Parameters:**
    - `shira_weight`: (num_shira_weight,) - 1D vector of trainable values
    - num_shira_weight = r × (in_features + out_features)
  - **Non-trainable Metadata:**
    - `shira_indices`: (2, num_shira_weight) - COO sparse tensor indices
    - `r`, `scaling`: Rank parameter and scaling factor per adapter
  - `adapter_layer_names = ("shira_weight",)`
  - `other_param_names = ("r", "scaling", "shira_indices")`
  - `update_layer()`: Initializes shira_weight (zeros or randn), extracts indices from mask
    - Converts binary mask to sparse COO indices via `torch.where(mask == 1.0)`
    - Validates indices match weight dimensions
  - `set_scale()`: Allows runtime scaling adjustment (default 1.0 during training)

- **Sparse Representation:**
  - Instead of storing full dense delta matrix, stores:
    1. Vector of trainable values (shira_weight)
    2. Fixed indices where values belong (shira_indices)
  - This avoids issues with torch.sparse_coo_tensor as nn.Parameter (PyTorch issue #79542)

- **Linear class** (concrete implementation):
  - Inherits from `nn.Module` and `ShiraLayer`
  - **forward()**: Three modes:
    1. `disable_adapters`: Run base layer only (unmerge if needed)
    2. `merged`: Base layer contains merged adapter
    3. Normal: Deep copies base weight, adds sparse delta, applies linear
      - Uses `F.linear(x, new_weight, bias)` with modified weight
  - **get_delta_weight()**: Constructs sparse COO tensor from indices and scaled weights
    - Returns: `torch.sparse_coo_tensor(indices, weights * scaling, shape)`
    - Handles multi-GPU by ensuring indices on correct device
  - **merge()**: Adds sparse delta directly to base layer weights
    - `base_layer.weight.data += get_delta_weight(adapter)`
  - **unmerge()**: Subtracts sparse delta from base layer weights

- **Design Choice:** Forward pass deep copies base weight and adds sparse delta
  - Alternative: Could use base_layer forward + sparse addition, but current approach uses F.linear for consistency

**Significance:** Core implementation of SHiRA's sparse high-rank adaptation. Key innovations:
1. **Sparse representation**: Only stores r(m+n) values + indices vs m×n full matrix
2. **Full rank**: Unlike LoRA's rank-r bottleneck, can update any pattern
3. **Same parameter count**: r(m+n) matches LoRA's 2×r×min(m,n) for typical r values
4. **Flexible masking**: Any sparsity pattern supported (random, structured, magnitude-based)
5. **Efficient storage**: Sparse COO tensor representation for merging
The method is particularly effective when adaptations require diverse patterns that low-rank factorization cannot capture well. The deep copy in forward pass has memory cost but ensures clean separation of base and adapter weights.
