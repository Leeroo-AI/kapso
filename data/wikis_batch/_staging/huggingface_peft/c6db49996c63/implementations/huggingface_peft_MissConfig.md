# MissConfig (MISS Configuration)

## Implementation Overview

**File:** `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/miss/config.py`
**Lines of Code:** 140
**Language:** Python

MissConfig is a dataclass configuration for MiSS (Minimally Structured Sparse) adapters, providing parameter-efficient fine-tuning through Householder reflections with three operational modes: standard balance, BAT (block-wise adaptive), and Mini (minimal parameters).

## Core Implementation

### Configuration Class

**Class:** `MissConfig(PeftConfig)`

Dataclass defining all configurable parameters for MiSS adapters:

```python
@dataclass
class MissConfig(PeftConfig):
    """
    Configuration class for MiSSModel.

    Args:
        r (`int`): Rank along in_features dimension (default: 64)
        miss_dropout (`float`): Dropout probability (default: 0.0)
        mini_r (`int`): Rank along out_features dimension (default: 1)
        target_modules (`Optional[Union[List[str], str]]`): Module names to adapt
        exclude_modules (`Optional[Union[List[str], str]]`): Module names to exclude
        init_weights (bool | Literal["bat", "mini"]): Initialization variant
        layers_to_transform (`Optional[Union[list[int], int]]`): Specific layers to adapt
        layers_pattern (`Optional[str]`): Layer pattern for selective adaptation
        bias (`str`): Bias handling - 'none', 'all', or 'MiSS_only'
        modules_to_save (`Optional[list[str]]`): Additional trainable modules
    """

    r: int = field(
        default=64,
        metadata={
            "help": "The rank of MiSS corresponds to a low-rank decomposition along the in_features dimension.",
            "note": "It is best to set 'r' to an even number; otherwise, the default initialization method will not work.",
        },
    )
    miss_dropout: float = field(default=0.0, metadata={"help": "MiSS dropout"})
    mini_r: int = field(
        default=1,
        metadata={
            "help": "The rank of MiSS corresponds to a low-rank decomposition along the out_features dimension.",
            "note": "It is recommended that mini_r be divisible by out_features. When mini_r == out_features, the mini method is equivalent to the default efficient MiSS.",
        },
    )
```

### Rank Configuration

**Primary Rank (r):**

Controls decomposition along input features:

```python
r: int = field(
    default=64,
    metadata={
        "help": "The rank of MiSS corresponds to a low-rank decomposition along the in_features dimension.",
        "note": "It is best to set 'r' to an even number; otherwise, the default initialization method will not work.",
    },
)
```

**Best Practices:**
- Use even numbers for r (e.g., 32, 64, 128)
- Higher r = more capacity but more parameters
- For 768-dim models: r=64 typically works well
- For larger models: r=128 or 256 may be needed

**Secondary Rank (mini_r):**

Controls decomposition along output features (Mini mode):

```python
mini_r: int = field(
    default=1,
    metadata={
        "help": "The rank of MiSS corresponds to a low-rank decomposition along the out_features dimension.",
        "note": "It is recommended that mini_r be divisible by out_features. When mini_r == out_features, the mini method is equivalent to the default efficient MiSS.",
    },
)
```

**Recommended Settings:**
- `mini_r=1`: Standard MiSS mode (default)
- `mini_r=out_features`: Equivalent to standard mode
- `mini_r=out_features//k`: k-fold parameter reduction
- Must satisfy: `out_features % mini_r == 0`

### Module Selection

**Target Modules:**

Specifies which layers to adapt:

```python
target_modules: Optional[Union[list[str], str]] = field(
    default=None,
    metadata={
        "help": "List of module names or regex expression of the module names to replace with MiSS.",
        "example": "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' ",
    },
)
```

**Examples:**
- `['q_proj', 'v_proj']`: Adapt query and value projections
- `'.*attn.*'`: All attention layers (regex)
- `'all-linear'`: All linear layers except output

**Exclude Modules:**

Prevents adaptation of specific layers:

```python
exclude_modules: Optional[Union[list[str], str]] = field(
    default=None,
    metadata={"help": "List of module names or regex expression of the module names to exclude from MiSS."},
)
```

### Initialization Modes

**Init Weights Parameter:**

Controls operational mode through initialization:

```python
init_weights: bool | Literal["bat", "mini"] = field(
    default=True,
    metadata={
        "help": (
            "True -> MiSS balance; `bat` -> Bat; `mini` -> smaller rank and efficiency"
            "Whether to initialize the weights of the MiSS layers with their default initialization. Don't change "
            "this setting, except if you know exactly what you're doing."
        ),
    },
)
```

**Mode Details:**

1. **Standard (True):**
   - Shape: `[r, out_features]`
   - Initialization: zeros
   - Most general and efficient

2. **BAT ("bat"):**
   - Shape: `[out_features // r, r, r]`
   - Initialization: zeros
   - Enables nonlinear block updates
   - Requires: `in_features % r == 0` and `out_features % r == 0`

3. **Mini ("mini"):**
   - Shape: `[r, mini_r]`
   - Initialization: zeros
   - Maximum parameter efficiency
   - Requires: `out_features % mini_r == 0`

### Layer Selection

**Selective Layer Adaptation:**

```python
layers_to_transform: Optional[Union[list[int], int]] = field(
    default=None,
    metadata={
        "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
    },
)
layers_pattern: Optional[str] = field(
    default=None,
    metadata={
        "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
    },
)
```

**Examples:**
- `layers_to_transform=[0, 5, 10]`: Only layers 0, 5, and 10
- `layers_to_transform=5`: Only layer 5
- `layers_pattern="decoder.layers"`: Custom layer pattern

### Bias Configuration

**Bias Handling:**

```python
bias: str = field(default="none", metadata={"help": "Bias type for MiSS. Can be 'none', 'all' or 'MiSS_only'"})
```

**Options:**
- `"none"`: No bias training (default)
- `"all"`: Train all biases
- `"MiSS_only"`: Only train MiSS adapter biases

### Additional Trainable Modules

**Modules to Save:**

```python
modules_to_save: Optional[list[str]] = field(
    default=None,
    metadata={
        "help": "List of modules apart from MiSS layers to be set as trainable and saved in the final checkpoint. "
        "For example, in Sequence Classification or Token Classification tasks, "
        "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
    },
)
```

**Common Use Cases:**
- Classification heads: `['classifier', 'score']`
- Token embeddings: `['embed_tokens']`
- Task-specific layers

## Post-Initialization Validation

### Configuration Validation

**Method:** `__post_init__()`

Validates and processes configuration after initialization:

```python
def __post_init__(self):
    super().__post_init__()
    self.peft_type = PeftType.MISS

    # Convert lists to sets for efficient lookup
    self.target_modules = (
        set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
    )
    self.exclude_modules = (
        set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
    )

    # Validate regex and layer transform compatibility
    if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
        raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

    if isinstance(self.target_modules, str) and self.layers_pattern is not None:
        raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")
```

**Validation Rules:**
1. Sets PEFT type to MISS
2. Converts module lists to sets for efficient matching
3. Prevents conflicting configurations (regex + layer indices)
4. Ensures mutually exclusive parameters don't coexist

## Configuration Examples

### Example 1: Standard MiSS (Attention-only)

```python
from peft import MissConfig

config = MissConfig(
    r=64,  # Even number recommended
    miss_dropout=0.1,
    target_modules=['q_proj', 'v_proj'],  # Query and value only
    init_weights=True,  # Standard mode
    bias="none"
)
```

**Parameters per layer:** 64 × out_features
**Use case:** Efficient fine-tuning with good performance

### Example 2: BAT Mode (All Linear)

```python
config = MissConfig(
    r=32,  # Must divide layer dimensions
    miss_dropout=0.0,
    target_modules='all-linear',
    init_weights="bat",  # Block-wise adaptive
    layers_to_transform=list(range(12, 24))  # Only layers 12-23
)
```

**Parameters per layer:** (out_features/32) × 32 × 32
**Use case:** Nonlinear updates for complex tasks

### Example 3: Mini Mode (Maximum Efficiency)

```python
config = MissConfig(
    r=64,
    mini_r=16,  # 64x parameter reduction vs standard
    miss_dropout=0.05,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    init_weights="mini",
    exclude_modules=['lm_head']
)
```

**Parameters per layer:** 64 × 16 = 1,024
**Use case:** Extreme parameter efficiency

### Example 4: Classification Task

```python
config = MissConfig(
    r=128,
    miss_dropout=0.1,
    target_modules='all-linear',
    init_weights=True,
    modules_to_save=['classifier'],  # Train classification head
    bias="all"  # Train all biases for classification
)
```

### Example 5: Regex Pattern Matching

```python
config = MissConfig(
    r=64,
    target_modules=r'.*decoder\.layers\.\d+\.self_attn\.(q|v)_proj$',  # Regex
    init_weights=True
)
# Cannot use layers_to_transform with regex!
```

## Parameter Efficiency Comparison

Given a layer with dimensions 768 × 768:

### Standard MiSS (r=64, mini_r=1)
- **Parameters:** 64 × 768 = 49,152
- **Reduction vs full:** 768²/49,152 = 12.0x

### BAT Mode (r=64)
- **Parameters:** (768/64) × 64 × 64 = 49,152
- **Structure:** Block-diagonal
- **Same count as standard but different behavior**

### Mini Mode (r=64, mini_r=16)
- **Parameters:** 64 × 16 = 1,024
- **Reduction vs standard:** 49,152/1,024 = 48x
- **Reduction vs full:** 768²/1,024 = 576x

### LoRA Comparison (r=64)
- **LoRA Parameters:** 2 × 768 × 64 = 98,304
- **MiSS Standard:** 49,152 (2x fewer)
- **MiSS Mini:** 1,024 (96x fewer!)

## Design Patterns

### Dataclass Pattern

Uses Python dataclasses for clean configuration:

```python
@dataclass
class MissConfig(PeftConfig):
    r: int = field(default=64, metadata={...})
    # Automatic __init__, __repr__, etc.
```

### Strategy Pattern

Different initialization modes via single parameter:

```python
init_weights: bool | Literal["bat", "mini"]
# True -> Standard
# "bat" -> BAT mode
# "mini" -> Mini mode
```

### Builder Pattern

Fluent configuration construction:

```python
config = MissConfig(
    r=64,
    miss_dropout=0.1,
    target_modules=['q_proj', 'v_proj']
)
model = get_peft_model(base_model, config)
```

## Integration Points

### With PeftConfig

Inherits base PEFT configuration:

```python
class MissConfig(PeftConfig):
    # Inherits: peft_type, task_type, inference_mode, etc.
```

### With MissModel

Configuration consumed by model:

```python
class MissModel(BaseTuner):
    def _create_and_replace(self, miss_config, ...):
        r = miss_config.r
        mini_r = miss_config.mini_r
        init_weights = miss_config.init_weights
        # Use configuration to create layers
```

## Validation Rules

### Automatic Validations

1. **Rank positivity:** r > 0 (checked in layer)
2. **Module format:** List converted to set
3. **Regex conflicts:** Cannot mix regex with layer indices

### Runtime Validations

Performed when creating layers:

1. **BAT mode dimensions:**
   ```python
   if self.in_features % r != 0 or self.out_features % r != 0:
       raise ValueError("The weight matrix must be fully divisible into [r, r] blocks.")
   ```

2. **Mini mode dimensions:**
   ```python
   if self.out_features % mini_r != 0:
       raise ValueError("out_features must be divisible by mini_r")
   ```

## Usage Patterns

### Basic Usage

```python
from peft import MissConfig, get_peft_model
from transformers import AutoModel

# Create configuration
config = MissConfig(
    r=64,
    target_modules=['q_proj', 'v_proj']
)

# Apply to model
base_model = AutoModel.from_pretrained("bert-base-uncased")
model = get_peft_model(base_model, config)

# Train
model.train()
```

### Multiple Adapter Usage

```python
# Base configuration
config1 = MissConfig(r=64, target_modules=['q_proj', 'v_proj'])
model = get_peft_model(base_model, config1, adapter_name="task1")

# Add second adapter
config2 = MissConfig(r=128, target_modules='all-linear')
model.add_adapter("task2", config2)

# Switch between adapters
model.set_adapter("task1")
model.set_adapter("task2")
```

### Serialization

```python
# Save configuration
config = MissConfig(r=64, target_modules=['q_proj', 'v_proj'])
model = get_peft_model(base_model, config)
model.save_pretrained("./miss_adapter")

# Load configuration
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "./miss_adapter")
```

## Best Practices

### Rank Selection

1. **Start with r=64** for most tasks
2. **Use even numbers** (32, 64, 128, 256)
3. **Scale with model size:**
   - Small models (BERT-base): r=32-64
   - Large models (LLaMA-7B): r=64-128
   - Very large models: r=128-256

### Mode Selection

1. **Standard mode (True):** Default choice, good balance
2. **BAT mode ("bat"):** When dimensions perfectly divisible by r
3. **Mini mode ("mini"):** When extreme parameter efficiency needed

### Module Targeting

1. **Attention-only:** Start with `['q_proj', 'v_proj']`
2. **All attention:** Add `['k_proj', 'o_proj']`
3. **All-linear:** Use for maximum adaptation
4. **Exclude heads:** Always exclude `['lm_head', 'classifier']`

### Dropout

1. **No dropout (0.0):** Inference or small datasets
2. **Light dropout (0.05-0.1):** Standard training
3. **Heavy dropout (0.2):** Preventing overfitting

## Common Pitfalls

### Dimension Mismatch (BAT Mode)

```python
# Wrong: dimensions not divisible
config = MissConfig(
    r=64,
    init_weights="bat"  # Error if layer dims not divisible by 64!
)

# Correct: check dimensions first
if in_features % 64 == 0 and out_features % 64 == 0:
    config = MissConfig(r=64, init_weights="bat")
else:
    config = MissConfig(r=64, init_weights=True)  # Use standard
```

### Mini Mode Parameter Mismatch

```python
# Wrong: out_features = 768, mini_r = 100 (not divisible)
config = MissConfig(r=64, mini_r=100, init_weights="mini")  # Error!

# Correct: use divisor
config = MissConfig(r=64, mini_r=96, init_weights="mini")  # 768 % 96 == 0
```

### Regex with Layer Indices

```python
# Wrong: mixing regex and layer indices
config = MissConfig(
    target_modules='.*attn.*',  # Regex
    layers_to_transform=[0, 1, 2]  # Error!
)

# Correct: use list or regex, not both
config = MissConfig(
    target_modules=['attn.q_proj', 'attn.v_proj'],
    layers_to_transform=[0, 1, 2]  # OK
)
```

## References

- **Paper**: "MiSS: Minimally Structured Sparse Parameter-Efficient Fine-Tuning" (2024)
- **URL**: https://huggingface.co/papers/2409.15371
- **Related**: LoRA, AdaLoRA configuration patterns
- **PeftType**: `PeftType.MISS`
