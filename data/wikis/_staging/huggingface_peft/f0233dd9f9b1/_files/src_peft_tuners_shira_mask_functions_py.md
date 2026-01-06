# File: `src/peft/tuners/shira/mask_functions.py`

**Category:** utility

| Property | Value |
|----------|-------|
| Lines | 73 |
| Functions | `random_mask` |
| Imports | torch, typing |

## Understanding

**Status:** ✅ Fully explored

**Purpose:** Provides mask generation functions for SHiRA adapters, defining which weight elements should be trainable.

**Mechanism:**
- **Module-level Documentation:**
  - Specifies required signature for mask functions:
    - Positional args: `base_layer` (nn.Module), `r` (int)
    - Keyword args: Optional, implementation-specific
    - Return: Binary torch.tensor (0/1) matching base_layer.weight shape and device/dtype
  - Documents pattern for custom mask functions with additional arguments
  - Provides example workflow for using custom masks with get_peft_model()

- **random_mask() function**: Default mask generator
  - **Algorithm:**
    1. Calculate total elements: `base_layer.weight.numel()`
    2. Calculate num_shira_weights: `r * (shape[0] + shape[1])`
    3. Generate random permutation of all indices
    4. Select first num_shira_weights indices
    5. Create mask with ones at selected indices, zeros elsewhere
  - **Parameters:**
    - `base_layer`: Linear layer to generate mask for
    - `r`: Parameter budget controller
    - `random_seed` (Optional): Seed for torch.Generator (deterministic masks)
  - **Implementation:**
    - Uses `torch.randperm()` with optional Generator for reproducibility
    - Flattens weight to 1D, selects indices, reshapes to original shape
    - Uses `scatter_()` to place ones at selected positions

- **Extensibility Pattern:**
  - Users can define custom mask functions with arbitrary logic
  - Example use cases:
    - Magnitude-based: Select largest weights by importance
    - Structured: Select entire rows/columns or blocks
    - Learned: Use pruning methods to identify important connections
    - Task-specific: Domain knowledge about which connections matter

**Significance:** Critical utility enabling SHiRA's flexibility. The mask function interface is the key to SHiRA's adaptability:
1. **Random baseline**: random_mask provides strong default performance
2. **Reproducibility**: random_seed ensures deterministic masks across runs
3. **Extensibility**: Clean interface for custom strategies
4. **Research**: Enables investigating different sparsity patterns
The random mask strategy is surprisingly effective - despite no learned structure, random sparsity often performs well. This suggests that parameter count matters more than specific sparsity pattern for many tasks. Advanced users can explore structured, magnitude-based, or learned masks for potentially better task-specific performance.
