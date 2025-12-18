# File: `src/peft/tuners/shira/config.py`

**Category:** configuration

| Property | Value |
|----------|-------|
| Lines | 130 |
| Classes | `ShiraConfig` |
| Imports | __future__, dataclasses, mask_functions, peft, typing, warnings |

## Understanding

**Status:** ✅ Fully explored

**Purpose:** Configuration class for SHiRA (Sparse High Rank Adapter), defining hyperparameters for sparse adaptation.

**Mechanism:**
- **ShiraConfig dataclass** extends `PeftConfig` with SHiRA-specific parameters:
  - `r` (int, default=32): Parameter budget controller - num_params = r(m+n) for m×n matrix (same as LoRA)
    - Unlike LoRA, this r doesn't restrict rank - SHiRA is high-rank
  - `mask_type` (Literal["random"], default="random"): Type of mask function
    - Default: random sparse mask
    - Custom masks supported via `config.mask_fn = custom_function`
  - `mask_fn`: Callable that generates binary mask (0/1) indicating which elements to train
    - Must return mask of shape (m, n) with exactly r(m+n) ones
    - Must match device and dtype of base layer
  - `random_seed` (Optional[int]): Seed for deterministic random mask generation
  - `target_modules`: Module names to apply SHiRA (only linear layers supported)
  - `fan_in_fan_out` (bool): True for layers storing weights as (fan_in, fan_out) like Conv1D
  - `init_weights` (bool, default=True): Initialize to zeros (False for random init, testing only)
  - `modules_to_save`: Additional modules to train and save

- **Validation in __post_init__:**
  - Sets `peft_type = PeftType.SHIRA`
  - Converts target_modules list to set
  - If mask_type=="random": assigns `mask_fn = random_mask`
  - If unrecognized mask_type: warns user to set custom mask_fn

**Significance:** Core configuration enabling SHiRA's sparse high-rank adaptation. The key innovation is using a sparse mask to select which weight elements to update, achieving LoRA parameter parity while maintaining full rank. This can capture more complex adaptations than low-rank methods. The mask_fn interface allows flexible masking strategies beyond random (e.g., magnitude-based, structured, learned). The r parameter controls sparsity level: for a 768×768 matrix, r=32 means only 32(768+768)=49,152 elements are trainable vs 589,824 total (8.3% sparsity).
