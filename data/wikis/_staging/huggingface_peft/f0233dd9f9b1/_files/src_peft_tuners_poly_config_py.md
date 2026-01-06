# File: `src/peft/tuners/poly/config.py`

**Category:** configuration

| Property | Value |
|----------|-------|
| Lines | 104 |
| Classes | `PolyConfig` |
| Imports | __future__, dataclasses, peft, typing |

## Understanding

**Status:** ✅ Fully explored

**Purpose:** Configuration class for Poly (Polytropon) adapter, which defines all hyperparameters and settings needed for multi-task learning with LoRA-based adapters.

**Mechanism:**
- **PolyConfig dataclass** extends `PeftConfig` with Poly-specific parameters:
  - `r` (int, default=8): LoRA attention dimension/rank for each skill
  - `target_modules`: Module names to apply Poly to (e.g., ['q', 'v'] for attention layers)
  - `exclude_modules`: Modules to explicitly exclude from Poly adaptation
  - `modules_to_save`: Additional modules to train and save (e.g., classification heads)
  - `init_weights` (bool, default=True): Whether to initialize Poly weights properly
  - `poly_type` (Literal["poly"], default="poly"): Variant of Poly to use (currently only "poly" supported)
  - `n_tasks` (int, default=1): Number of tasks in multitasking scenario
  - `n_skills` (int, default=4): Number of LoRA modules (skills) per Poly layer
  - `n_splits` (int, default=1): Number of splits within each LoRA for Multi-Head Routing (MHR)
- Sets `peft_type = PeftType.POLY` in `__post_init__`
- Converts module lists to sets for efficient lookup

**Significance:** Core configuration that enables Poly's multi-task learning capabilities. Poly combines multiple LoRA adapters with learned routing weights, allowing efficient parameter sharing across tasks. The `n_skills` parameter controls how many LoRA modules are available, while `n_tasks` determines routing capacity. When `n_splits > 1`, it enables Multi-Head Routing (MHR) for more fine-grained control. Referenced in papers: Polytropon (2022) and Multi-Head Routing (2021).
