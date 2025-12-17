# File: `src/peft/tuners/shira/mask_functions.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 72 |
| Functions | `random_mask` |
| Imports | torch, typing |

## Understanding

**Status:** ✅ Documented

**Purpose:** Provides mask generation functions for SHiRA, defining which weight elements will be trainable. Contains the default random_mask implementation and documentation for custom mask functions.

**Mechanism:** The random_mask function takes base_layer, r, and optional random_seed, computes num_shira_weights = r*(shape[0] + shape[1]), uses torch.randperm to select random indices, and returns a binary mask (1s at selected positions, 0s elsewhere). Extensive module docstring explains required signature (base_layer, r, **kwargs), return format (binary tensor matching base_layer.weight shape and dtype/device), and pattern for creating custom mask functions via closures.

**Significance:** Extensibility point for SHiRA allowing custom sparsity patterns. The random_mask provides a strong baseline, but users can implement structured sparsity (e.g., block-sparse, importance-based) by defining custom functions. Complete examples in examples/shira/ directory. The careful documentation enables experimentation with different sparsity structures for optimal performance.
