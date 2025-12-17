# File: `src/peft/tuners/poly/router.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 81 |
| Classes | `Router`, `PolyRouter` |
| Functions | `get_router` |
| Imports | abc, config, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements routing mechanisms for Poly that determine task-specific skill mixing weights.

**Mechanism:** Defines abstract Router base class. PolyRouter maintains module_logits (n_tasks × n_splits×n_skills) parameter initialized near zero (uniform [-1e-3, 1e-3]). Forward pass indexes logits by task_ids, reshapes to (bs, n_splits, n_skills), applies RelaxedBernoulli sampling during training (for differentiability) or sigmoid during inference, then normalizes to sum to 1 per split: module_weights = module_logits / (sum + EPS). get_router() factory function instantiates appropriate router based on poly_type.

**Significance:** Implements Poly's learned routing mechanism that determines how to combine the n_skills LoRA modules for each task. The learnable module_logits enable the model to discover which skill combinations work best for each task during training. RelaxedBernoulli provides differentiable sampling for gradient-based learning. The normalization ensures mixing weights are valid probability distributions, allowing interpretable skill attribution per task.
