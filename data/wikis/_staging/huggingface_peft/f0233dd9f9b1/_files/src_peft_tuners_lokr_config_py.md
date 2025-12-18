# src/peft/tuners/lokr/config.py

## Overview
Configuration class for LoKr (Low-Rank Kronecker Product) adaptation, extending `LycorisConfig` with LoKr-specific parameters. This configuration controls how Kronecker product factorization is applied to neural network weights for parameter-efficient fine-tuning.

## Class: LoKrConfig

Inherits from `LycorisConfig` and adds LoKr-specific behavior.

### Core Parameters

#### Rank Configuration
- **r** (int, default=8): LoKr rank for low-rank factorization
  - Lower values: Fewer parameters, less expressiveness
  - Higher values: More parameters, more expressiveness
- **alpha** (int, default=8): LoKr alpha parameter for scaling
  - Controls the scaling factor applied to LoKr updates
  - Scaling factor = alpha / r

#### Dropout Settings
- **rank_dropout** (float, default=0.0): Dropout probability for rank dimension during training
  - Applied to weight updates, not activations
  - Helps prevent overfitting
- **module_dropout** (float, default=0.0): Probability of disabling entire LoKr modules during training
  - Stochastic depth-style regularization
  - Module either fully active or fully disabled per forward pass
- **rank_dropout_scale** (bool, default=False): Whether to scale rank dropout during training
  - If True: Compensates for dropped dimensions by scaling remaining ones

#### Decomposition Options
- **decompose_both** (bool, default=False): Whether to perform rank decomposition of left Kronecker product matrix
  - If False: Only right matrix is decomposed
  - If True: Both matrices use low-rank factorization (more parameter efficient)
- **decompose_factor** (int, default=-1): Kronecker product decomposition factor
  - Controls how dimensions are factorized
  - -1 uses automatic factorization near square root
- **use_effective_conv2d** (bool, default=False): Use parameter-efficient decomposition for Conv2d/Conv1d with kernel_size > 1
  - Implements "Proposition 3" from FedPara paper
  - Only beneficial for convolutional layers with spatial kernels
  - Automatically disabled for 1x1 convolutions (no benefit)

#### Target Module Selection
- **target_modules** (Optional[Union[list[str], str]], default=None):
  - List of module names or regex patterns to apply LoKr
  - Examples: `['q', 'v']`, `'.*decoder.*(SelfAttention).*(q|v)$'`
  - Special value `'all-linear'`: Targets all linear/Conv1D layers except output layer
  - If None: Uses architecture-specific defaults from model mapping
- **exclude_modules** (Optional[Union[list[str], str]], default=None):
  - Module names or regex patterns to exclude from LoKr
  - Takes precedence over target_modules

#### Layer Selection
- **layers_to_transform** (Optional[Union[list[int], int]], default=None):
  - Specific layer indices to transform
  - Can be single integer or list of integers
  - Must specify `layers_pattern` when using this
- **layers_pattern** (Optional[Union[list[str], str]], default=None):
  - Pattern to identify the layer container (e.g., 'layers', 'h')
  - Required when `layers_to_transform` is specified
  - Points to the nn.ModuleList containing layers

#### Advanced Configuration
- **rank_pattern** (dict): Mapping from layer names/regex to custom ranks
  - Example: `{'^model.decoder.layers.0.encoder_attn.k_proj': 16}`
  - Allows per-layer rank customization
- **alpha_pattern** (dict): Mapping from layer names/regex to custom alphas
  - Same structure as rank_pattern
  - Enables per-layer scaling customization

#### Weight Initialization
- **init_weights** (Union[bool, Literal["lycoris"]], default=True):
  - True: Use standard initialization (recommended)
  - False: Random initialization (not recommended)
  - "lycoris": Initialize in LyCORIS repository style
  - Should generally not be changed unless you know what you're doing

#### Training Configuration
- **modules_to_save** (Optional[list[str]], default=None):
  - Additional modules to be trainable and saved
  - Useful for task-specific heads (e.g., classifier layers)
  - Common in classification/tagging tasks where final layer is randomly initialized

### Post-Initialization

The `__post_init__` method:
1. Calls parent's post-init
2. Sets `peft_type` to `PeftType.LOKR`
3. Converts target_modules to set if it's a list
4. Converts exclude_modules to set if it's a list
5. Validates that `layers_pattern` is specified when `layers_to_transform` is used

## Kronecker Product Decomposition

### What is LoKr?
LoKr decomposes weight updates using Kronecker products:
- Original weight: W (size m × n)
- LoKr decomposition: W ≈ (W1 ⊗ W2) * (alpha/r)
- Where ⊗ is the Kronecker product
- W1 and W2 can be further decomposed using low-rank factorization

### Parameter Efficiency
For a weight matrix of size (m × n):
- Full rank: m × n parameters
- LoKr with factorization (m1, m2) × (n1, n2) and rank r:
  - Basic: m1×n1 + m2×n2 parameters (if one side decomposed)
  - Advanced: m1×r + r×n1 + m2×r + r×n2 parameters (if both decomposed)
- Significant reduction when m and n are large

### When to Use LoKr vs LoRA
- **LoKr**: Better for structured matrix updates where Kronecker structure aligns with weight semantics
- **LoRA**: Better for general-purpose adaptation
- **LoKr**: Can be more parameter-efficient for very large matrices
- **LoRA**: Simpler, more widely tested

## Usage Patterns

### Basic Usage
```python
config = LoKrConfig(
    r=8,
    alpha=8,
    target_modules=["q_proj", "v_proj"],
    rank_dropout=0.1,
)
```

### Advanced Usage
```python
config = LoKrConfig(
    r=8,
    alpha=16,
    target_modules='all-linear',
    decompose_both=True,
    use_effective_conv2d=True,  # For vision models
    rank_pattern={'^model.decoder.layers.0': 16},  # Custom rank for layer 0
)
```

## Integration
This configuration is typically passed to `get_peft_model()` or used directly with `LoKrModel` to apply LoKr adaptation to a base model.
