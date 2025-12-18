# File: `src/peft/tuners/miss/config.py`

**Category:** Configuration

| Property | Value |
|----------|-------|
| Lines | 140 |
| Classes | `MissConfig` |
| Imports | __future__, dataclasses, peft, typing |

## Understanding

**Status:** Fully explored

**Purpose:** Defines `MissConfig` class that stores configuration parameters for MiSS (Mixture of Subspaces) adapter models using Householder reflection adaptation.

**Mechanism:**

### Core Parameters:
- **r** (int, default=64): Rank along in_features dimension (low-rank decomposition). Best set to even numbers for default initialization
- **miss_dropout** (float, default=0.0): Dropout probability for MiSS layers
- **mini_r** (int, default=1): Rank along out_features dimension. When `init_weights="mini"`, out_features should be divisible by mini_r
- **init_weights** (bool | Literal["bat", "mini"], default=True):
  - `True` (default): MiSS balance - most efficient and general method
  - `"bat"`: Enables nonlinear updates across different shards
  - `"mini"`: Smaller rank mode with fewer trainable parameters (requires out_features % mini_r == 0)

### Module Selection:
- **target_modules**: Modules to apply adapter (regex, list, or 'all-linear')
- **exclude_modules**: Modules to exclude from adaptation
- **layers_to_transform**: Specific layer indices to transform
- **layers_pattern**: Pattern for nn.ModuleList layers
- **bias** (str, default="none"): Bias handling - 'none', 'all', or 'MiSS_only'
- **modules_to_save**: Additional modules to train and save

### Validation in `__post_init__`:
1. Sets `peft_type` to `PeftType.MISS`
2. Converts target_modules and exclude_modules to sets if lists
3. Validates that `layers_to_transform` is None when target_modules is a regex string
4. Validates that `layers_pattern` is None when target_modules is a regex string

**Significance:** MissConfig controls the Householder reflection-based adaptation method. Unlike LoRA which uses explicit low-rank matrices, MiSS uses matrix-free Householder reflections to create orthogonal transformations efficiently. The three initialization modes (balance, bat, mini) provide flexibility in trading off parameter count, expressiveness, and computational efficiency. This approach can achieve competitive results with fewer parameters than traditional low-rank methods.

## Key Features

- **Dual-Rank Control**: Separate ranks for input (r) and output (mini_r) dimensions
- **Three Initialization Modes**: Balance (default), bat (nonlinear), mini (efficient)
- **Householder Reflections**: Matrix-free orthogonal transformations
- **Flexible Bias Handling**: None, all, or adapter-only bias training
- **Even Rank Recommendation**: Best results when r is even for default init

## References

- Paper: https://huggingface.co/papers/2409.15371
- Method: Householder reflection adaptation
