# File: `src/peft/tuners/loha/config.py`

**Category:** Configuration

| Property | Value |
|----------|-------|
| Lines | 143 |
| Classes | `LoHaConfig` |
| Imports | __future__, dataclasses, peft, typing |

## Understanding

**Status:** Fully explored

**Purpose:** Defines the configuration class `LoHaConfig` that stores all hyperparameters and settings for LoHa (Low-Rank Hadamard Product) adapter models.

**Mechanism:**

The `LoHaConfig` class extends `LycorisConfig` and provides comprehensive configuration for LoHa adapters:

### Core Parameters:
- **r** (int, default=8): LoHa rank - the dimensionality of the low-rank decomposition
- **alpha** (int, default=8): LoHa scaling factor (scaling = alpha / r)
- **rank_dropout** (float, default=0.0): Dropout probability for rank dimension during training
- **module_dropout** (float, default=0.0): Dropout probability for disabling entire LoHa modules during training
- **use_effective_conv2d** (bool, default=False): Enable parameter-efficient decomposition for Conv2d layers with kernel size > 1 (based on "Proposition 3" from FedPara paper)

### Module Selection:
- **target_modules**: Specifies which modules to adapt (supports regex, list, or 'all-linear')
- **exclude_modules**: Modules to explicitly exclude from adaptation
- **layers_to_transform**: Specific layer indices to transform
- **layers_pattern**: Pattern to match nn.ModuleList (e.g., 'layers', 'h')

### Advanced Features:
- **rank_pattern** (dict): Maps layer names/regex to custom ranks different from default
- **alpha_pattern** (dict): Maps layer names/regex to custom alpha values
- **modules_to_save**: Additional modules to train and save (e.g., classification heads)
- **init_weights** (bool, default=True): Whether to initialize adapter weights properly

### Validation in `__post_init__`:
1. Sets `peft_type` to `PeftType.LOHA`
2. Converts `target_modules` and `exclude_modules` to sets if they're lists
3. Validates that `layers_pattern` is only specified when `layers_to_transform` is also provided

**Significance:** This configuration class is the central control point for LoHa adapters. LoHa uses Hadamard products (element-wise multiplication) of two low-rank decompositions to achieve parameter-efficient fine-tuning with potentially better expressiveness than standard LoRA. The configuration supports fine-grained control over which layers to adapt, how to initialize parameters, and specialized optimizations for convolutional layers. The pattern-based rank and alpha customization enables layer-specific tuning strategies.

## Key Features

- **Inherits from LycorisConfig**: Part of the LyCORIS family of adapters
- **Dual Dropout Support**: Both rank-level and module-level dropout for regularization
- **Pattern-Based Customization**: Different ranks/alphas for different layers
- **Conv2d Optimization**: Special handling for convolutional layers with large kernels
- **Mixed Adapter Compatible**: Can be combined with other adapter types
