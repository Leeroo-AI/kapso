# FourierFTConfig (FourierFT Configuration)

## Implementation Overview

**File:** `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/fourierft/config.py`
**Lines of Code:** 206
**Language:** Python

FourierFTConfig defines parameters for FourierFT adapters that learn sparse spectral coefficients in the frequency domain for extreme parameter efficiency.

## Core Configuration

```python
@dataclass
class FourierFTConfig(PeftConfig):
    """Configuration for FourierFT

    Paper: https://huggingface.co/papers/2405.03003

    Key Parameters:
        n_frequency (int): Number of learnable frequencies (default: 1000)
        scaling (float): Scaling coefficient (default: 150.0)
        random_loc_seed (int): Seed for frequency selection (default: 777)
        target_modules: Modules to adapt
        n_frequency_pattern (dict): Per-layer frequency counts
        init_weights (bool): Initialize to zeros (default: False)
    """

    n_frequency: int = field(
        default=1000,
        metadata={
            "help": "Number of learnable frequencies. Higher = more capacity but more memory. "
                    "Must be in range (0, out_features * in_features]."
        }
    )
    scaling: float = field(
        default=150.0,
        metadata={
            "help": "Scaling coefficient (similar to lora_alpha). "
                    "Typical values: 100-150 for BERT/RoBERTa, 300 for LLaMA/ViT."
        }
    )
    random_loc_seed: Optional[int] = field(
        default=777,
        metadata={"help": "Seed for random frequency location selection"}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set to True for Conv1D layers (e.g., GPT-2)"}
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "Module names or regex to adapt. Only linear layers supported."}
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "Module names or regex to exclude"}
    )
    bias: str = field(
        default="none",
        metadata={"help": "Bias type: 'none', 'all', or 'fourier_only'"}
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Additional trainable modules"}
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={"help": "Specific layer indices to transform"}
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "Layer pattern for selective transformation"}
    )
    n_frequency_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={"help": "Per-layer frequency counts, e.g., {'model.decoder.layers.0.self_attn.q_proj': 500}"}
    )
    init_weights: bool = field(
        default=False,
        metadata={
            "help": "Initialize spectrum to zeros (True) or normal distribution (False, default)"
        }
    )
```

## Key Parameters

### Frequency Count (n_frequency)

```python
n_frequency: int = field(default=1000)
```

**Meaning:** Number of trainable spectral coefficients

**Impact:**
- Higher n_frequency → More capacity, more memory
- Lower n_frequency → Less capacity, more efficiency
- Directly equals parameter count per layer

**Constraints:**
- Must be > 0
- Must be ≤ out_features × in_features

**Recommendations by Task:**

**NLU (RoBERTa-large):**
```python
n_frequency=1000  # Comparable to LoRA r=8
# LoRA(r=8) = 16K params vs FourierFT = 1K params (16x fewer!)
```

**Image Classification (ViT-large):**
```python
n_frequency=3000  # Comparable to LoRA r=16
# LoRA(r=16) = 32K params vs FourierFT = 3K params (11x fewer!)
```

**LLM Fine-tuning:**
```python
n_frequency=2000-5000  # Depends on model size and task complexity
```

### Scaling Factor

```python
scaling: float = field(default=150.0)
```

**Meaning:** Multiplier for weight delta (similar to lora_alpha)

**Recommendations:**

**RoBERTa (base/large), BERT:**
```python
scaling=100.0  # or 150.0
```

**LLaMA family:**
```python
scaling=300.0
```

**ViT (base/large):**
```python
scaling=300.0
```

**Guidelines:**
- Start with recommended values
- Reduce if training unstable
- Increase for stronger adaptation
- Can tune via hyperparameter search

### Random Seed

```python
random_loc_seed: Optional[int] = field(default=777)
```

**Purpose:** Deterministic frequency selection

**Usage:**
- Same seed → Same frequency locations
- Enables reproducibility
- Important for checkpointing

### Per-Layer Frequency Patterns

```python
n_frequency_pattern: Optional[dict] = field(default_factory=dict)
```

**Purpose:** Override default n_frequency for specific layers

**Example:**
```python
config = FourierFTConfig(
    n_frequency=1000,  # Default
    n_frequency_pattern={
        'model.decoder.layers.0.encoder_attn.k_proj': 500,  # Lower for this layer
        'model.decoder.layers.23.self_attn.v_proj': 2000,   # Higher for this layer
    }
)
```

**Use Cases:**
- Lower frequencies for early layers
- Higher frequencies for later layers
- Task-specific adaptation patterns

### Initialization

```python
init_weights: bool = field(default=False)
```

**Options:**
- `False` (default): Random normal ~ N(0, 1)
- `True`: All zeros (no adaptation initially)

**Recommendation:**
- Use `False` for most cases
- Use `True` for gradual adaptation

## Validation

**Method:** `__post_init__()`

```python
def __post_init__(self):
    super().__post_init__()
    self.peft_type = PeftType.FOURIERFT

    self.target_modules = (
        set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
    )
    self.exclude_modules = (
        set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
    )

    # Validate incompatible options
    if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
        raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

    if isinstance(self.target_modules, str) and self.layers_pattern is not None:
        raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")

    if self.layers_pattern and not self.layers_to_transform:
        raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified.")
```

## Configuration Examples

### Example 1: NLU (RoBERTa)

```python
config = FourierFTConfig(
    n_frequency=1000,
    scaling=150.0,
    target_modules=['query', 'value'],
    init_weights=False
)
```

**Results:** Similar to LoRA(r=8) with 16x fewer parameters

### Example 2: Vision (ViT)

```python
config = FourierFTConfig(
    n_frequency=3000,
    scaling=300.0,
    target_modules='all-linear',
    exclude_modules=['head'],
    init_weights=False
)
```

**Results:** Similar to LoRA(r=16) with 11x fewer parameters

### Example 3: LLM (LLaMA)

```python
config = FourierFTConfig(
    n_frequency=2000,
    scaling=300.0,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    random_loc_seed=42
)
```

### Example 4: Per-Layer Frequencies

```python
config = FourierFTConfig(
    n_frequency=1000,  # Default
    scaling=150.0,
    target_modules='all-linear',
    n_frequency_pattern={
        # Lower frequencies for early layers
        'encoder.layer.0.attention.self.query': 500,
        'encoder.layer.0.attention.self.value': 500,
        # Higher frequencies for late layers
        'encoder.layer.11.attention.self.query': 2000,
        'encoder.layer.11.attention.self.value': 2000,
    }
)
```

### Example 5: Classification Task

```python
config = FourierFTConfig(
    n_frequency=1000,
    scaling=150.0,
    target_modules='all-linear',
    modules_to_save=['classifier'],
    bias='all',  # Train all biases for classification
    init_weights=False
)
```

## Parameter Efficiency Analysis

### Comparison with LoRA

For 768-dimensional layers (typical BERT):

| Method | Params per Layer | Ratio |
|--------|-----------------|-------|
| LoRA (r=4) | 6,144 | 6.1x |
| LoRA (r=8) | 12,288 | 12.3x |
| LoRA (r=16) | 24,576 | 24.6x |
| **FourierFT (n=1000)** | **1,000** | **1x** |

For 1024-dimensional layers (RoBERTa-large, ViT):

| Method | Params per Layer | Ratio |
|--------|-----------------|-------|
| LoRA (r=8) | 16,384 | 16.4x |
| LoRA (r=16) | 32,768 | 32.8x |
| **FourierFT (n=1000)** | **1,000** | **1x** |
| **FourierFT (n=3000)** | **3,000** | **3x** |

### Scaling Relationship

**Rule of thumb:**
```
FourierFT(n) ≈ LoRA(r) when:
n ≈ 2 * d * r / 16

Example:
d=768, LoRA r=8 → n = 2*768*8/16 = 768
But empirically, n=1000 works better!
```

## Performance Guidelines

### Choosing n_frequency

**Start with:**
- Small models (BERT-base): 500-1000
- Medium models (RoBERTa-large): 1000-2000
- Large models (LLaMA-7B): 2000-5000
- Very large models (LLaMA-70B): 5000-10000

**Tuning:**
- Increase if underfitting
- Decrease if overfitting or memory constrained
- Test in range [base/2, base*2]

### Choosing Scaling

**Default values:**
- BERT/RoBERTa: 100-150
- LLaMA: 300
- ViT: 300

**Adjustment:**
- Training unstable → Reduce by 50%
- Weak adaptation → Increase by 50-100%
- Fine-tune if critical

## Common Pitfalls

### Too Many Frequencies

```python
# Wrong: Exceeds weight dimensions
config = FourierFTConfig(
    n_frequency=1000000  # For 768x768 layer: max = 589,824
)  # Error!

# Correct:
config = FourierFTConfig(
    n_frequency=1000  # Well within limits
)
```

### Mixing Regex and Layer Transforms

```python
# Wrong: Incompatible options
config = FourierFTConfig(
    target_modules='.*attn.*',  # Regex
    layers_to_transform=[0, 1, 2]  # Error!
)

# Correct: Use one or the other
config = FourierFTConfig(
    target_modules=['attn.q_proj', 'attn.v_proj'],
    layers_to_transform=[0, 1, 2]  # OK
)
```

### Missing layers_to_transform

```python
# Wrong: layers_pattern without layers_to_transform
config = FourierFTConfig(
    layers_pattern='decoder.layers'  # Error!
)

# Correct:
config = FourierFTConfig(
    layers_pattern='decoder.layers',
    layers_to_transform=list(range(12))
)
```

## References

- **Paper**: https://huggingface.co/papers/2405.03003
- **Type**: `PeftType.FOURIERFT`
- **Key Innovation**: Sparse spectral learning
- **Efficiency**: 10-20x fewer parameters than LoRA
