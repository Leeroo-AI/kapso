# Implementation: loraplus.py

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/optimizers/loraplus.py`
- **Size**: 121 lines
- **Module**: `peft.optimizers.loraplus`
- **Description**: LoRA+ learning rate scheduling for improved training efficiency

## Overview

LoRA+ is an optimizer configuration that applies different learning rates to LoRA A and B matrices, enabling faster convergence and better performance. The key insight is that B matrices (which directly affect output) should be updated more aggressively than A matrices (which are often randomly initialized).

**Reference**: [LoRA+ Paper](https://huggingface.co/papers/2402.12354)

## Core Function: create_loraplus_optimizer

```python
def create_loraplus_optimizer(
    model: PeftModel,
    optimizer_cls: type[Optimizer],
    *,
    lr: float,
    loraplus_lr_ratio: float,
    **kwargs
) -> Optimizer:
    """
    Creates optimizer with LoRA+-specific parameter grouping.

    Parameter Groups:
    - groupA: lora_A parameters (lr = lr)
    - groupB: lora_B and biases (lr = lr * loraplus_lr_ratio)
    - groupB_no_decay: lora_B without weight decay
    - embedding: Embedding layer LoRA (lr = loraplus_lr_embedding)

    Args:
        model: PeftModel with LoRA adapters
        optimizer_cls: Optimizer class (e.g., torch.optim.AdamW)
        lr: Base learning rate (ηA)
        loraplus_lr_ratio: Ratio ηB/ηA (typically 2.0-16.0)
        loraplus_weight_decay: Weight decay for LoRA parameters
        loraplus_lr_embedding: Learning rate for embedding LoRA

    Returns:
        Configured optimizer with parameter groups
    """
```

## Parameter Grouping Strategy

### Group Classification

**groupA** (Base LR):
```python
if "lora_B" not in name and param.ndim > 1:
    param_groups["groupA"][name] = param
```
- LoRA A matrices
- Initialized randomly (Kaiming/Xavier)
- Lower learning rate for stability

**groupB** (High LR with decay):
```python
if "lora_B" in name or param.ndim == 1:
    if name in decay_parameters:
        param_groups["groupB"][name] = param
```
- LoRA B matrices
- Biases
- Initialized to zero (often)
- Higher learning rate for faster adaptation

**groupB_no_decay** (High LR, no decay):
```python
if "lora_B" in name or param.ndim == 1:
    if name not in decay_parameters:
        param_groups["groupB_no_decay"][name] = param
```
- LoRA B in layers without decay
- Prevents over-regularization

**embedding** (Custom LR):
```python
if isinstance(module, Embedding):
    param_groups["embedding"][name] = param
```
- Embedding layer LoRA parameters
- Often need special treatment
- Default: 1e-6 (very small)

### Optimizer Configuration

```python
optimizer_grouped_parameters = [
    {
        "params": list(param_groups["groupA"].values()),
        "weight_decay": loraplus_weight_decay,
        "lr": lr,  # Base learning rate
    },
    {
        "params": list(param_groups["embedding"].values()),
        "weight_decay": loraplus_weight_decay,
        "lr": loraplus_lr_embedding,  # Very small (default 1e-6)
    },
    {
        "params": list(param_groups["groupB"].values()),
        "weight_decay": loraplus_weight_decay,
        "lr": lr * loraplus_lr_ratio,  # High learning rate
    },
    {
        "params": list(param_groups["groupB_no_decay"].values()),
        "weight_decay": 0.0,
        "lr": lr * loraplus_lr_ratio,  # High LR, no decay
    },
]

optimizer = optimizer_cls(optimizer_grouped_parameters, **kwargs)
```

## Key Features

### Differential Learning Rates

**Motivation**:
- A: Random init → needs careful exploration
- B: Zero init → can be aggressive

**Typical Ratios**:
- Easy tasks: 2-4×
- Medium tasks: 4-8×
- Hard tasks: 8-16×

**Tuning Guidance**:
```python
# Harder task = larger ratio + smaller base LR
lr = 5e-5  # (vs typical 1e-4)
loraplus_lr_ratio = 16.0  # (vs typical 8.0)
```

### Weight Decay Handling

**LayerNorm Exclusion**:
```python
decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
decay_parameters = [name for name in decay_parameters if "bias" not in name]
```

**Per-Group Decay**:
- groupA/B: Apply weight decay
- groupB_no_decay: No decay (biases, norms)

### 8-bit Optimization Support

```python
eight_bit_names = ["Adam8bit", "AdamW8bit", "PagedAdam8bit", "PagedAdamW8bit"]
if optimizer_cls.__name__ in eight_bit_names:
    import bitsandbytes
    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            manager.register_module_override(module, "weight", {"optim_bits": 32})
```

**Purpose**: Keep embedding updates in FP32 for stability

## Usage Examples

### Basic Usage

```python
from peft import get_peft_model, LoraConfig
from peft.optimizers import create_loraplus_optimizer
from torch.optim import AdamW

config = LoraConfig(r=64, lora_alpha=16)
model = get_peft_model(base_model, config)

optimizer = create_loraplus_optimizer(
    model,
    optimizer_cls=AdamW,
    lr=5e-5,
    loraplus_lr_ratio=16.0,
    loraplus_weight_decay=0.01
)
```

### With 8-bit Optimizer

```python
import bitsandbytes as bnb

optimizer = create_loraplus_optimizer(
    model,
    optimizer_cls=bnb.optim.AdamW8bit,
    lr=5e-5,
    loraplus_lr_ratio=8.0
)
```

### Custom Embedding LR

```python
optimizer = create_loraplus_optimizer(
    model,
    optimizer_cls=AdamW,
    lr=5e-5,
    loraplus_lr_ratio=8.0,
    loraplus_lr_embedding=5e-6  # 10× smaller than base
)
```

### Task-Specific Configuration

**Easy Task** (high base LR, low ratio):
```python
optimizer = create_loraplus_optimizer(
    model, AdamW,
    lr=2e-4,
    loraplus_lr_ratio=2.0
)
```

**Hard Task** (low base LR, high ratio):
```python
optimizer = create_loraplus_optimizer(
    model, AdamW,
    lr=2e-5,
    loraplus_lr_ratio=16.0
)
```

## Performance Characteristics

### Convergence Speed

**Typical Speedup**:
- 1.5-2× faster convergence
- Fewer training steps to reach target performance
- Better final performance in many cases

**Why It Works**:
- B matrices learn faster (zero init → optimal value)
- A matrices explore carefully (random init → avoid instability)

### Memory Usage

**Identical to Standard Optimizer**:
- Same number of optimizer states
- Same gradient computation
- Only learning rates differ

### Stability

**Improved Stability vs Uniform High LR**:
- High LR on B: Fast adaptation
- Low LR on A: Prevents divergence
- Net effect: Fast + stable

## Best Practices

### Learning Rate Selection

1. **Start Conservative**: lr=5e-5, ratio=8.0
2. **Increase Ratio**: If convergence slow, try 12-16
3. **Decrease Base LR**: If unstable, halve lr (keep ratio)
4. **Tune Together**: Smaller lr ↔ larger ratio

### Embedding Handling

**When to Use Custom Embedding LR**:
- Token-level fine-tuning
- Vocabulary expansion
- Sparse token updates

**Default (1e-6) works when**:
- No LoRA on embeddings
- Embeddings frozen
- Large vocabulary (prevents overfitting)

### Weight Decay

**Typical Values**:
- Standard: 0.01
- Small models: 0.001
- Large models: 0.01-0.1
- No decay on biases/norms (automatic)

## Comparison

| Method | ηA | ηB | Convergence | Stability |
|--------|----|----|-------------|-----------|
| Standard LoRA | η | η | Baseline | Baseline |
| LoRA+ | η | 8η-16η | 1.5-2× faster | Same/Better |
| High Uniform | 8η | 8η | Fast | Unstable |
| Low Uniform | η/8 | η/8 | Slow | Stable |

## Limitations

1. **Hyperparameter Sensitivity**: Ratio requires tuning
2. **LoRA-Specific**: Only applicable to LoRA adapters
3. **No Theoretical Guarantee**: Empirical method
4. **Embedding LR**: May need task-specific tuning

## Implementation Notes

### Parameter Detection

**Using operator.attrgetter**:
```python
from operator import attrgetter

module = attrgetter(name)(model)  # Efficient nested attribute access
if isinstance(module, Embedding):
    # ...
```

**Benefits**:
- Handles nested attributes (e.g., "model.layer.0.proj")
- More efficient than recursive getattr
- Type checking on actual module

### Bias/LayerNorm Detection

```python
decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
decay_parameters = [name for name in decay_parameters if "bias" not in name]
```

**Excluded from Decay**:
- All biases
- LayerNorm weights
- RMSNorm weights (in ALL_LAYERNORM_LAYERS)

## Cross-References

- **Paper**: [LoRA+: Efficient Low Rank Adaptation](https://huggingface.co/papers/2402.12354)
- **Reference**: [Original Implementation](https://github.com/nikhil-ghosh-berkeley/loraplus/)
- **Related**: `peft.optimizers.lorafa`, `peft.tuners.lora`
- **Dependencies**: `torch.optim`, `transformers.trainer_pt_utils`, `bitsandbytes` (optional)
