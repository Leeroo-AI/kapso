# LoHa Tuner Implementation

## Metadata
- **Type**: PEFT Tuner Implementation
- **Module Path**: `src/peft/tuners/loha/`
- **Components**: Layer, Config, Model
- **Lines of Code**: 703 (layer: 444, config: 143, model: 116)
- **PEFT Type**: `PeftType.LOHA`
- **Paper**: https://huggingface.co/papers/2108.06098
- **Reference**: https://github.com/KohakuBlueleaf/LyCORIS

## Overview

LoHa (Low-Rank Hadamard Product) is a parameter-efficient fine-tuning method that decomposes weight updates using Hadamard products of low-rank matrices. By combining two low-rank decompositions elementwise, LoHa achieves better expressiveness than standard LoRA with similar parameter counts.

**Key Features**:
- Hadamard product of two low-rank decompositions: `(W1_a @ W1_b) ⊙ (W2_a @ W2_b)`
- Optional CP decomposition for convolutional layers
- Rank and module dropout for regularization
- Supports Linear, Conv1d, and Conv2d layers
- Parameter-effective Conv2d decomposition

## Core Components

### 1. LoHaLayer (`layer.py`)

Base adapter layer implementing Hadamard product low-rank decomposition.

**Key Classes**:
- `LoHaLayer`: Base tuner layer from LycorisLayer
- `Linear`: LoHa implementation for Linear layers
- `Conv2d`: LoHa implementation for 2D convolutional layers
- `Conv1d`: LoHa implementation for 1D convolutional layers

**Adapter Parameters**:
```python
adapter_layer_names = (
    "hada_w1_a", "hada_w1_b",  # First decomposition
    "hada_w2_a", "hada_w2_b",  # Second decomposition
    "hada_t1", "hada_t2"        # CP tensors (for Conv)
)
```

**State Management**:
- `hada_w1_a`, `hada_w1_b`: First low-rank decomposition (A @ B)
- `hada_w2_a`, `hada_w2_b`: Second low-rank decomposition (A @ B)
- `hada_t1`, `hada_t2`: CP decomposition tensors for convolutions
- `r`, `alpha`, `scaling`: Per-adapter hyperparameters
- `rank_dropout`, `module_dropout`: Regularization parameters

### 2. LoHaConfig (`config.py`)

Configuration dataclass for LoHa adapters, extending LycorisConfig.

**Key Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `r` | int | 8 | LoHa rank for decomposition |
| `alpha` | int | 8 | Scaling factor (scale = alpha / r) |
| `rank_dropout` | float | 0.0 | Dropout on rank dimension during training |
| `module_dropout` | float | 0.0 | Dropout for disabling entire modules |
| `use_effective_conv2d` | bool | False | Use CP decomposition for Conv2d (ksize > 1) |
| `target_modules` | Optional[Union[list[str], str]] | None | Module names or regex to apply LoHa |
| `exclude_modules` | Optional[Union[list[str], str]] | None | Module names or regex to exclude |
| `init_weights` | bool | True | Initialize to identity (A random, B zero) |
| `layers_to_transform` | Optional[Union[list[int], int]] | None | Specific layer indices |
| `layers_pattern` | Optional[Union[list[str], str]] | None | Layer pattern for selection |
| `rank_pattern` | dict | {} | Per-layer rank overrides |
| `alpha_pattern` | dict | {} | Per-layer alpha overrides |
| `modules_to_save` | Optional[list[str]] | None | Additional trainable modules |

**Validation**:
- Requires `layers_to_transform` when `layers_pattern` is specified
- Converts `target_modules` and `exclude_modules` to sets if lists

### 3. LoHaModel (`model.py`)

Model wrapper that injects LoHa adapters into pretrained models.

**Class Attributes**:
- `prefix`: "hada_"
- `tuner_layer_cls`: `LoHaLayer`
- `target_module_mapping`: `TRANSFORMERS_MODELS_TO_LOHA_TARGET_MODULES_MAPPING`
- `layers_mapping`: Dict mapping layer types to LoHa implementations

## Implementation Details

### Hadamard Product Decomposition

The core idea is to decompose weight updates as:

```
ΔW = (W1_a @ W1_b) ⊙ (W2_a @ W2_b) × (alpha / r)
```

Where `⊙` denotes elementwise (Hadamard) product.

**For Linear layers**:
```python
# Parameter shapes
hada_w1_a: [out_features, r]
hada_w1_b: [r, in_features]
hada_w2_a: [out_features, r]
hada_w2_b: [r, in_features]

# Compute delta weight
delta_weight = (hada_w1_a @ hada_w1_b) * (hada_w2_a @ hada_w2_b)
delta_weight = delta_weight * (alpha / r)
```

**For Conv2d (standard)**:
```python
# Flatten spatial dimensions
shape = (out_channels, in_channels * kernel_h * kernel_w)

# Apply same Hadamard product decomposition
delta_weight = make_weight(w1_a, w1_b, w2_a, w2_b, scale)
delta_weight = delta_weight.reshape(out_channels, in_channels, kernel_h, kernel_w)
```

**For Conv2d (effective, use_effective_conv2d=True)**:
Uses CP (Canonical Polyadic) decomposition for parameter efficiency:

```python
# Additional tensors for spatial dimensions
hada_t1: [r, r, kernel_h, kernel_w]
hada_t2: [r, r, kernel_h, kernel_w]
hada_w1_a: [r, out_channels]
hada_w1_b: [r, in_channels]
hada_w2_a: [r, out_channels]
hada_w2_b: [r, in_channels]

# Reconstruct using einsum operations
rebuild1 = einsum('ijkl, jr, ip -> prkl', hada_t1, hada_w1_b, hada_w1_a)
rebuild2 = einsum('ijkl, jr, ip -> prkl', hada_t2, hada_w2_b, hada_w2_a)
delta_weight = rebuild1 * rebuild2 * scale
```

### Custom Autograd Functions

LoHa implements custom backward passes for memory efficiency:

**HadaWeight** (Linear/Conv flattened):
```python
class HadaWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w1a, w1b, w2a, w2b, scale):
        ctx.save_for_backward(w1a, w1b, w2a, w2b, scale)
        diff_weight = ((w1a @ w1b) * (w2a @ w2b)) * scale
        return diff_weight

    @staticmethod
    def backward(ctx, grad_out):
        (w1a, w1b, w2a, w2b, scale) = ctx.saved_tensors
        grad_out = grad_out * scale

        # Efficient gradient computation
        temp = grad_out * (w2a @ w2b)
        grad_w1a = temp @ w1b.T
        grad_w1b = w1a.T @ temp

        temp = grad_out * (w1a @ w1b)
        grad_w2a = temp @ w2b.T
        grad_w2b = w2a.T @ temp

        return grad_w1a, grad_w1b, grad_w2a, grad_w2b, None
```

**HadaWeightCP** (Conv with CP decomposition):
Similar structure but uses `einsum` operations for 4D tensors.

### Initialization Strategy

**Default Initialization** (`init_weights=True`):
```python
# First decomposition: random + zeros
nn.init.kaiming_uniform_(hada_w1_a, a=math.sqrt(5))
nn.init.kaiming_uniform_(hada_w1_b, a=math.sqrt(5))

# Second decomposition: random + zeros
nn.init.kaiming_uniform_(hada_w2_a, a=math.sqrt(5))
nn.init.zeros_(hada_w2_b)  # Zero initialization for identity

# CP tensors (if present)
nn.init.kaiming_uniform_(hada_t1, a=math.sqrt(5))
nn.init.kaiming_uniform_(hada_t2, a=math.sqrt(5))
```

**Random Initialization** (`init_weights=False`):
```python
# All matrices random (for testing/experimentation)
nn.init.kaiming_uniform_(hada_w1_a, a=math.sqrt(5))
nn.init.kaiming_uniform_(hada_w1_b, a=math.sqrt(5))
nn.init.kaiming_uniform_(hada_w2_a, a=math.sqrt(5))
nn.init.kaiming_uniform_(hada_w2_b, a=math.sqrt(5))
```

### Regularization Mechanisms

**Rank Dropout**:
Drops entire rows of the delta weight during training:

```python
if self.training and rank_dropout > 0:
    # Create dropout mask for output dimension
    drop = (torch.rand(weight.size(0)) > rank_dropout).to(weight.dtype)
    drop = drop.view(-1, *[1] * len(weight.shape[1:])).to(weight.device)

    # Scale to maintain expected value
    drop /= drop.mean()

    # Apply dropout
    weight *= drop
```

**Module Dropout**:
Randomly disables entire adapter modules during training:

```python
if self.training and torch.rand(1) > module_dropout:
    # Apply adapter
    result = result + delta_activations
else:
    # Skip adapter completely
    pass
```

### Forward Pass Logic

**Linear Forward**:
```python
def forward(self, x):
    # Base layer output
    result = self.base_layer(x)

    # Apply active adapters
    for adapter in active_adapters:
        if not_dropped(module_dropout):
            # Compute delta weight with rank dropout
            delta_weight = get_delta_weight(adapter)

            # Apply as additional linear transformation
            result = result + F.linear(x, delta_weight)

    return result
```

**Conv2d Forward**:
```python
def forward(self, x):
    result = self.base_layer(x)

    for adapter in active_adapters:
        if not_dropped(module_dropout):
            delta_weight = get_delta_weight(adapter)

            # Apply as additional convolution
            result = result + F.conv2d(
                x, delta_weight,
                stride=base_layer.stride,
                padding=base_layer.padding,
                dilation=base_layer.dilation,
                groups=base_layer.groups
            )

    return result
```

## I/O Contract

### Input
- **x**: `torch.Tensor` - Input tensor
  - Linear: `[batch, ..., in_features]`
  - Conv2d: `[batch, in_channels, height, width]`
  - Conv1d: `[batch, in_channels, length]`
- **args/kwargs**: Additional arguments passed to base layer

### Output
- **result**: `torch.Tensor` - Output with adapter contributions
- Maintains input dtype through transformations

### Constraints
- Rank `r` should be positive integer
- For Conv2d with `use_effective_conv2d=True`: kernel_size > 1
- Kernel_size=1 automatically disables effective_conv2d for efficiency
- Rank/module dropout only active during training
- Supports safe merge with NaN detection

## Usage Examples

### Basic LoHa Adapter
```python
from peft import LoHaConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure LoHa
config = LoHaConfig(
    r=8,
    alpha=8,
    target_modules=["q_proj", "v_proj"],
    rank_dropout=0.0,
    module_dropout=0.0,
    init_weights=True
)

# Apply LoHa adapter
peft_model = get_peft_model(model, config)
print(f"Trainable parameters: {peft_model.num_parameters(only_trainable=True)}")
```

### With Regularization
```python
# Use dropout for regularization
config = LoHaConfig(
    r=8,
    alpha=16,  # Higher alpha for more aggressive updates
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    rank_dropout=0.1,    # Drop 10% of rank dimensions
    module_dropout=0.05,  # Disable modules 5% of time
    init_weights=True
)

peft_model = get_peft_model(model, config)
```

### Diffusion Model with Effective Conv2d
```python
from diffusers import StableDiffusionPipeline
from peft import LoHaModel, LoHaConfig

# Configure for text encoder
config_te = LoHaConfig(
    r=8,
    alpha=32,
    target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
    rank_dropout=0.0,
    module_dropout=0.0,
    init_weights=True,
)

# Configure for U-Net with effective Conv2d
config_unet = LoHaConfig(
    r=8,
    alpha=32,
    target_modules=[
        "proj_in", "proj_out", "to_k", "to_q", "to_v",
        "to_out.0", "ff.net.0.proj", "ff.net.2",
    ],
    rank_dropout=0.0,
    module_dropout=0.0,
    init_weights=True,
    use_effective_conv2d=True,  # Use CP decomposition for Conv2d
)

# Load and adapt pipeline
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline.text_encoder = LoHaModel(pipeline.text_encoder, config_te, "default")
pipeline.unet = LoHaModel(pipeline.unet, config_unet, "default")
```

### Per-Layer Rank Patterns
```python
# Different ranks for different layers
config = LoHaConfig(
    r=8,  # Default rank
    alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    rank_pattern={
        "^model.layers.0.": 16,  # Higher rank for first layer
        "^model.layers.[1-5].": 8,  # Medium rank for layers 1-5
        "^model.layers.[6-9].": 4,  # Lower rank for layers 6-9
    },
    alpha_pattern={
        "^model.layers.0.": 32,  # Matching alpha scaling
        "^model.layers.[1-5].": 16,
        "^model.layers.[6-9].": 8,
    },
    init_weights=True
)

peft_model = get_peft_model(model, config)
```

### Computer Vision Application
```python
from transformers import ViTForImageClassification

# Load vision model
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

# LoHa for attention layers
config = LoHaConfig(
    r=8,
    alpha=16,
    target_modules=["query", "value"],
    rank_dropout=0.1,
    module_dropout=0.05,
    init_weights=True
)

peft_model = get_peft_model(model, config)
```

### Training and Merging
```python
# Train the model
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()

# Merge adapters for inference
peft_model.merge_adapter()

# Save merged model
peft_model.save_pretrained("./loha_merged_model")

# Or save adapter weights only
peft_model.save_pretrained("./loha_adapter", safe_serialization=True)
```

## Performance Characteristics

### Parameter Count
For rank `r`:
- **Linear**: `2 × (out_features × r + r × in_features)` = `4 × r × d` (average)
- **Conv2d (standard)**: `4 × r × (out_ch + in_ch × k²)`
- **Conv2d (effective)**: `4 × r × (out_ch + in_ch) + 2 × r² × k²`

**Compared to LoRA**:
- 2× more parameters for same rank
- Higher expressiveness due to Hadamard product
- Better performance on some tasks

### Memory Usage
- Forward pass requires computing and storing Hadamard products
- Custom autograd functions reduce backward pass memory
- Rank dropout adds minimal overhead
- Module dropout has no memory overhead

### Computational Complexity
- **Forward**: O(r × d²) - two low-rank products + elementwise multiply
- **Backward**: Optimized via custom autograd functions
- **Overhead vs LoRA**: ~1.5-2× due to double decomposition
- **Effective Conv2d**: Reduces computation for large kernels

### Training Characteristics
- Good convergence properties
- Rank dropout improves generalization
- Module dropout reduces overfitting
- Effective for both NLP and vision tasks

## Comparison with Other Methods

| Method | Parameters | Expressiveness | Conv Support | Dropout |
|--------|------------|----------------|--------------|---------|
| LoHa | 4rd | Higher | ✓ (+ CP) | Rank + Module |
| LoRA | 2rd | Medium | ✓ | Module only |
| LoKr | 3rd | Medium | ✓ | Module only |
| AdaLoRA | Variable | Medium | ✗ | Module only |

## Related Pages
- [[LoRA Tuner Implementation]] - Standard low-rank adaptation
- [[LoKr Tuner Implementation]] - Kronecker product low-rank
- [[LyCORIS Methods]] - Family of low-rank methods
- [[Adapter Dropout Strategies]] - Dropout techniques for PEFT
- [[PEFT Configuration Guide]] - General PEFT configuration patterns
- [[Convolutional Layer Adaptation]] - Conv-specific PEFT techniques
