# File: `src/peft/tuners/loha/config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 143 |
| Classes | `LoHaConfig` |
| Imports | __future__, dataclasses, peft, typing |

## Understanding

**Status:** ✅ Documented

**Purpose:** Configuration dataclass for LoHa (Low-Rank Hadamard Product) adapters, defining all hyperparameters and settings needed to create and train LoHa layers.

**Mechanism:** Extends LycorisConfig with LoHa-specific parameters including rank (r), alpha scaling factor, rank_dropout, module_dropout, and use_effective_conv2d for efficient convolution decomposition. Validates configuration in __post_init__ by setting peft_type to LOHA and ensuring consistency between layers_pattern and layers_to_transform parameters.

**Significance:** Central configuration class that controls LoHa adapter behavior. The use_effective_conv2d flag enables parameter-efficient decomposition for convolutional layers as described in the FedPara paper. This configuration enables fine-grained control over which modules to adapt and how they behave during training.
