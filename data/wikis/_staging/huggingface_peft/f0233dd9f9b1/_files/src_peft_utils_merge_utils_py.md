# File: `src/peft/utils/merge_utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 268 |
| Functions | `reshape_weight_task_tensors`, `magnitude_based_pruning`, `random_pruning`, `prune`, `calculate_majority_sign_mask`, `disjoint_merge`, `task_arithmetic`, `magnitude_prune`, `... +3 more` |
| Imports | torch, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements various adapter merging strategies for combining multiple PEFT adapters into a single model, supporting task arithmetic, pruning, and sign-based conflict resolution.

**Mechanism:** Provides 6 merging algorithms: task_arithmetic (weighted sum), magnitude_prune (top-k by magnitude), ties (prune + elect majority sign + disjoint merge), dare_linear/dare_ties (random dropout + rescale). Includes utilities for pruning (magnitude/random), sign mask calculation, and disjoint merging. All operations work on stacked tensors with proper weight reshaping.

**Significance:** Enables model soup and multi-adapter fusion techniques from recent research. Critical for scenarios like merging task-specific adapters or creating unified models from multiple LoRA experts. Implements algorithms from papers like TIES-Merging and DARE for parameter-efficient model combination.
