# File: `src/peft/tuners/randlora/config.py`

**Category:** configuration

| Property | Value |
|----------|-------|
| Lines | 200 |
| Classes | `RandLoraConfig` |
| Imports | dataclasses, peft, typing, warnings |

## Understanding

**Status:** ✅ Fully explored

**Purpose:** Configuration class for RandLora adapter, defining hyperparameters for random basis projection with trainable scaling.

**Mechanism:**
- **RandLoraConfig dataclass** extends `PeftConfig` with RandLora-specific parameters:
  - `r` (int, default=32): Random basis rank dimension (inversely proportional to trainable parameters - smaller r means MORE parameters)
  - `target_modules`: Module names to apply RandLora (only linear layers supported)
  - `projection_prng_key` (int, default=0): PRNG seed for reproducible random basis initialization
  - `save_projection` (bool, default=True): Whether to save basis_A/basis_B in checkpoint (increases size but ensures cross-system compatibility)
  - `sparse` (bool, default=False): Use ternary sparse bases {-1, 0, 1} with 1/6 probability for ±1, 2/3 for 0
  - `very_sparse` (bool, default=False): Use highly sparse bases with attribution probability 1/√D
  - `randlora_dropout` (float): Dropout probability for RandLora layers
  - `randlora_alpha` (float, default=640): Scaling coefficient (typically 20x the rank, but large values can cause instability)
  - `fan_in_fan_out` (bool): Set True for layers storing weights as (fan_in, fan_out) like Conv1D
  - `bias` (str): Bias type - 'none', 'all', or 'randlora_only'
  - `modules_to_save`: Additional modules to train and save
  - `init_weights` (bool, default=True): Initialize with default initialization
  - `layers_to_transform`: Specific layer indices to transform
  - `layers_pattern`: Layer pattern name for layer-specific transformation

- **Validation:**
  - Sets `peft_type = PeftType.RANDLORA`
  - Converts target_modules list to set for efficient lookup
  - Warns if `save_projection=False` about potential restoration issues

**Significance:** Core configuration enabling RandLora's memory-efficient design. Unlike LoRA where trainable parameters scale with rank, RandLora uses frozen random projections with trainable diagonal scaling matrices (lambda/gamma). The `r` parameter is inverse to traditional LoRA - smaller r means larger random bases are used, requiring more trainable scaling parameters. The sparse options enable future matmul-free computation optimizations. Paper reference: https://huggingface.co/papers/2502.00987.
