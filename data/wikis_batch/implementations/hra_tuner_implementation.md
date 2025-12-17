# HRA Tuner Implementation

## Metadata
- **Type**: PEFT Tuner Implementation
- **Module Path**: `src/peft/tuners/hra/`
- **Components**: Layer, Config, Model
- **Lines of Code**: 725 (layer: 461, config: 133, model: 131)
- **PEFT Type**: `PeftType.HRA`
- **Paper**: https://huggingface.co/papers/2405.17484

## Overview

HRA (Householder Reflection Adaptation) is a parameter-efficient fine-tuning method that applies orthogonal transformations to model weights using Householder reflections. Instead of adding low-rank updates like LoRA, HRA multiplies weights by orthogonal matrices constructed from Householder reflections, preserving weight norms and structure.

**Key Features**:
- Orthogonal weight transformations via Householder reflections
- Preserves weight norm and structure
- Optional Gram-Schmidt orthogonalization for numerical stability
- Supports both Linear and Conv2d layers
- Symmetric initialization for even ranks

## Core Components

### 1. HRALayer (`layer.py`)

Base adapter layer implementing Householder reflection transformations.

**Key Classes**:
- `HRALayer`: Base tuner layer with orthogonal adapter management
- `HRALinear`: HRA implementation for Linear layers
- `HRAConv2d`: HRA implementation for 2D convolutional layers

**Adapter Parameters**:
```python
adapter_layer_names = ("hra_u",)
other_param_names = ("hra_r", "hra_apply_GS")
```

**State Management**:
- `hra_u`: ParameterDict storing Householder vectors
  - Linear: `[in_features, r]`
  - Conv2d: `[in_channels × kernel_h × kernel_w, r]`
- `hra_r`: Dictionary mapping adapter names to rank values
- `hra_apply_GS`: Dictionary indicating whether to use Gram-Schmidt
- `merged_adapters`: List of currently merged adapters

### 2. HRAConfig (`config.py`)

Configuration dataclass for HRA adapters.

**Key Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `r` | int | 8 | Number of Householder reflections (prefer even) |
| `apply_GS` | bool | False | Apply Gram-Schmidt orthogonalization |
| `target_modules` | Optional[Union[list[str], str]] | None | Module names or regex to apply HRA |
| `exclude_modules` | Optional[Union[list[str], str]] | None | Module names or regex to exclude |
| `init_weights` | bool | True | Use symmetric initialization |
| `layers_to_transform` | Optional[Union[list[int], int]] | None | Specific layer indices to transform |
| `layers_pattern` | Optional[Union[list[str], str]] | None | Layer pattern for selective transformation |
| `bias` | str | "none" | Bias handling: 'none', 'all', 'hra_only' |
| `modules_to_save` | Optional[list[str]] | None | Additional modules to train/save |

**Validation**:
- Requires `layers_to_transform` when `layers_pattern` is specified
- Prevents `layers_to_transform`/`layers_pattern` with regex `target_modules`

### 3. HRAModel (`model.py`)

Model wrapper that injects HRA adapters into pretrained models.

**Class Attributes**:
- `prefix`: "hra_"
- `tuner_layer_cls`: `HRALayer`
- `target_module_mapping`: `TRANSFORMERS_MODELS_TO_HRA_TARGET_MODULES_MAPPING`

## Implementation Details

### Householder Reflection Theory

A Householder reflection is an orthogonal transformation defined by:
```
H = I - 2uu^T   where u is a unit vector
```

Multiple reflections are composed:
```
W_new = W_base × H_1 × H_2 × ... × H_r
```

**Properties**:
- Orthogonal: `H^T H = I`
- Norm-preserving: `||Hx|| = ||x||`
- Symmetric: `H = H^T`
- Involutory: `H^2 = I` (self-inverse)

### Orthogonalization Methods

#### Standard Method (apply_GS=False)
Applies reflections sequentially with normalized vectors:

```python
# Normalize each Householder vector
opt_u = opt_u / opt_u.norm(dim=0)  # [features, r]

# Compose reflections iteratively
weight = torch.eye(in_features)
for i in range(r):
    ui = opt_u[:, i].view(-1, 1)
    weight = weight - 2 * weight @ ui @ ui.t()

# Final transformation
W_new = W_base @ weight
```

#### Gram-Schmidt Method (apply_GS=True)
Explicitly orthogonalizes Householder vectors:

```python
# Orthogonalize using modified Gram-Schmidt
weight = [(opt_u[:, 0] / opt_u[:, 0].norm()).view(-1, 1)]
for i in range(1, r):
    ui = opt_u[:, i].view(-1, 1)
    # Remove projections onto previous vectors
    for j in range(i):
        ui = ui - (weight[j].t() @ ui) * weight[j]
    weight.append((ui / ui.norm()).view(-1, 1))

# Stack and create reflection matrix
weight = torch.cat(weight, dim=1)
weight = I - 2 * weight @ weight.t()
```

### Initialization Strategy

**Symmetric Initialization** (default, even ranks):
```python
# For even r, create symmetric pairs
if r % 2 == 0:
    half_u = torch.zeros(in_features, r // 2)
    nn.init.kaiming_uniform_(half_u, a=math.sqrt(5))
    # Repeat each column twice for symmetry
    self.hra_u = nn.Parameter(torch.repeat_interleave(half_u, 2, dim=1))
```

**Random Initialization** (odd ranks or init_weights=False):
```python
nn.init.kaiming_uniform_(self.hra_u, a=math.sqrt(5))
```

**Symmetric Initialization Benefits**:
- Improves training stability
- Better gradient flow in early training
- Recommended for most applications

### Forward Pass Logic

**HRALinear Forward**:
```python
if not disable_adapters and not merged:
    # Start with identity
    new_weight = torch.eye(in_features, device=x.device)

    # Apply all active adapters
    for adapter in active_adapters:
        delta_weight = get_delta_weight(adapter)
        new_weight = new_weight @ delta_weight

    # Transform base weights
    final_weight = base_weight @ new_weight

    # Apply linear transformation
    result = F.linear(x, final_weight, bias)
```

**HRAConv2d Forward**:
```python
# Reshape conv weights to 2D
orig_weight = base_weight.view(
    out_channels,
    in_channels × kernel_h × kernel_w
)

# Apply Householder transformations
final_weight = orig_weight @ householder_matrix

# Reshape back and apply convolution
final_weight = final_weight.view(
    out_channels, in_channels, kernel_h, kernel_w
)
result = F.conv2d(x, final_weight, bias, stride, padding)
```

### Merge/Unmerge Operations

**Merge Process**:
```python
# Compute Householder transformation matrix
delta_weight = get_delta_weight(adapter_name)

# Right-multiply base weights
new_weight = base_weight @ delta_weight
base_layer.weight.data = new_weight
```

**Unmerge Process**:
```python
# Compute inverse transformation (reverse order)
delta_weight = get_delta_weight(adapter_name, reverse=True)

# Apply inverse transformation
original_weight = merged_weight @ delta_weight
base_layer.weight.data = original_weight
```

**Note**: Unmerging is exact due to orthogonality (`H^{-1} = H`)

## I/O Contract

### Input
- **x**: `torch.Tensor` - Input tensor
  - Linear: `[batch, ..., in_features]`
  - Conv2d: `[batch, in_channels, height, width]`
- **args/kwargs**: Additional arguments passed to base layer

### Output
- **result**: `torch.Tensor` - Transformed output
  - Linear: `[batch, ..., out_features]`
  - Conv2d: `[batch, out_channels, height', width']`
- Maintains input dtype through transformations

### Constraints
- Rank `r` preferably even for symmetric initialization
- Householder vectors have same dimensionality as weight columns
- Supports safe merge with NaN detection
- Exact unmerge due to orthogonality
- Scaling operations not supported (warns and sets scale to 1)

## Usage Examples

### Basic HRA Adapter
```python
from peft import HRAConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure HRA
config = HRAConfig(
    r=8,  # Even number recommended
    apply_GS=False,
    target_modules=["q_proj", "v_proj"],
    init_weights=True
)

# Apply HRA adapter
peft_model = get_peft_model(model, config)
print(f"Trainable parameters: {peft_model.num_parameters(only_trainable=True)}")
```

### With Gram-Schmidt Orthogonalization
```python
# Use Gram-Schmidt for numerical stability
config = HRAConfig(
    r=8,
    apply_GS=True,  # Explicit orthogonalization
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    init_weights=True
)

peft_model = get_peft_model(model, config)
```

### Diffusion Model Application
```python
from diffusers import StableDiffusionPipeline
from peft import HRAModel, HRAConfig

# Configure for text encoder
config_te = HRAConfig(
    r=8,
    target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
    init_weights=True,
)

# Configure for U-Net (includes Conv2d layers)
config_unet = HRAConfig(
    r=8,
    target_modules=[
        "proj_in", "proj_out", "to_k", "to_q", "to_v",
        "to_out.0", "ff.net.0.proj", "ff.net.2",
    ],
    init_weights=True,
)

# Load and adapt pipeline
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline.text_encoder = HRAModel(pipeline.text_encoder, config_te, "default")
pipeline.unet = HRAModel(pipeline.unet, config_unet, "default")
```

### Selective Layer Application
```python
# Apply only to specific layers
config = HRAConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    layers_to_transform=[0, 1, 2, 3, 4],  # First 5 layers
    layers_pattern="model.layers",
    apply_GS=False
)

peft_model = get_peft_model(model, config)
```

### Training and Merging
```python
# Train the model
trainer.train()

# Merge adapters (exact due to orthogonality)
peft_model.merge_adapter()

# Save merged model
peft_model.save_pretrained("./hra_merged_model")

# Unmerge is exact (no approximation error)
peft_model.unmerge_adapter()

# Continue training or save adapter only
peft_model.save_pretrained("./hra_adapter", safe_serialization=True)
```

### Computer Vision Application
```python
from transformers import ViTForImageClassification

# Load vision model
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

# HRA works well for both Linear and Conv2d layers
config = HRAConfig(
    r=8,
    target_modules=["query", "value"],  # Attention projections
    apply_GS=True,
    init_weights=True
)

peft_model = get_peft_model(model, config)
```

## Performance Characteristics

### Parameter Efficiency
- **Parameters per layer**: `in_features × r`
- **Compared to LoRA**: Similar for same rank `r`
- **Memory footprint**: Comparable to LoRA

### Computational Complexity
- **Forward pass**: O(in_features² × r) for weight transformation
- **Standard method**: Sequential application of reflections
- **Gram-Schmidt method**: Slightly slower but more numerically stable
- **Merge/Unmerge**: Exact operations (no approximation error)

### Numerical Properties
- **Norm preservation**: Maintains weight magnitudes
- **Stability**: Gram-Schmidt improves conditioning
- **Gradient flow**: Good due to orthogonality
- **Symmetric init**: Improves early training stability

## Comparison with Other Methods

| Aspect | HRA | LoRA | Adapter |
|--------|-----|------|---------|
| Parameters | r × d | 2 × r × d | hidden × d |
| Weight change | Orthogonal transform | Additive update | Bottleneck |
| Norm preservation | ✓ | ✗ | ✗ |
| Exact unmerge | ✓ | ✓ | N/A |
| Numerical stability | Good (with GS) | Good | Good |

## Related Pages
- [[LoRA Tuner Implementation]] - Additive low-rank adaptation
- [[BONE Tuner Implementation]] - Block-wise affine transforms
- [[Adapter Tuner Implementation]] - Bottleneck adapter layers
- [[Orthogonal Weight Methods]] - Other orthogonal approaches
- [[PEFT Configuration Guide]] - General PEFT configuration patterns
