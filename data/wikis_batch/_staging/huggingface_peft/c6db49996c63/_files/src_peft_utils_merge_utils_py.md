# File: `src/peft/utils/merge_utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 268 |
| Functions | `reshape_weight_task_tensors`, `magnitude_based_pruning`, `random_pruning`, `prune`, `calculate_majority_sign_mask`, `disjoint_merge`, `task_arithmetic`, `magnitude_prune`, `... +3 more` |
| Imports | torch, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides advanced techniques for merging multiple task-specific adapters into unified models.

**Mechanism:** Implements various merging strategies including task arithmetic (adding/subtracting task vectors), TIES (resolving parameter conflicts via magnitude-based pruning and sign consensus), DARE (random pruning), and disjoint merge (no parameter sharing). Handles weight tensor reshaping for compatibility.

**Significance:** Enables multi-task learning and model composition by intelligently combining multiple fine-tuned adapters, allowing a single model to perform multiple tasks without catastrophic forgetting or parameter conflicts.
