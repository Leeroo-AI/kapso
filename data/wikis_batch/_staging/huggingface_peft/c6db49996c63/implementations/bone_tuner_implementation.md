# BONE Tuner Implementation

## Metadata
- **Type**: PEFT Tuner Implementation
- **Module Path**: `src/peft/tuners/bone/`
- **Components**: Layer, Config, Model
- **Lines of Code**: 607 (layer: 352, config: 129, model: 126)
- **PEFT Type**: `PeftType.BONE`

## Overview

BONE (Block-wise Affine Transform) is a parameter-efficient fine-tuning method that applies block-wise affine transformations to model weights. The implementation provides two variants:
- **BONE**: Block-wise affine adaptation using shared blocks across the input dimension
- **BAT**: Block-wise affine transform with full divisibility requirements

**Note**: BONE will be removed in v0.19.0 of PEFT and replaced with `MissConfig`. A conversion script is available at `/scripts/convert-bone-to-miss.py`.

## Core Components

### 1. BoneLayer (`layer.py`)

Base adapter layer implementing block-wise affine transformations.

**Key Classes**:
- `BoneLayer`: Base tuner layer with adapter management
- `BoneLinear`: BONE implementation for dense (Linear) layers

**Adapter Parameters**:
```python
adapter_layer_names = ("bone_block",)
other_param_names = ("bone_r",)
```

**State Management**:
- `bone_r`: Dictionary mapping adapter names to rank values
- `bone_block`: ParameterDict storing block transformation parameters
- `merged_adapters`: List of currently merged adapters
- `_disable_adapters`: Flag to enable/disable all adapters

### 2. BoneConfig (`config.py`)

Configuration dataclass for BONE adapters.

**Key Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `r` | int | 64 | Rank of BONE blocks (prefer even numbers) |
| `target_modules` | Optional[Union[list[str], str]] | None | Module names or regex to apply BONE |
| `exclude_modules` | Optional[Union[list[str], str]] | None | Module names or regex to exclude |
| `init_weights` | bool \| Literal["bat"] | True | Initialization method (True=BONE, "bat"=BAT) |
| `layers_to_transform` | Optional[Union[list[int], int]] | None | Specific layer indices to transform |
| `layers_pattern` | Optional[str] | None | Layer pattern for selective transformation |
| `bias` | str | "none" | Bias handling: 'none', 'all', 'bone_only' |
| `modules_to_save` | Optional[list[str]] | None | Additional modules to train/save |

**Validation**:
- Converts `target_modules` and `exclude_modules` to sets if provided as lists
- Prevents `layers_to_transform` and `layers_pattern` when `target_modules` is regex
- Warns about deprecation in favor of `MissConfig`

### 3. BoneModel (`model.py`)

Model wrapper that injects BONE adapters into pretrained models.

**Class Attributes**:
- `prefix`: "bone_"
- `tuner_layer_cls`: `BoneLayer`
- `target_module_mapping`: `TRANSFORMERS_MODELS_TO_BONE_TARGET_MODULES_MAPPING`

## Implementation Details

### Weight Transformation Methods

#### BONE Method (Default)
Applies block-wise additive transformations:
```python
# Forward pass
bone = self.bone_block[adapter]  # Shape: [r, out_features]
r = bone.size(0)
# Reshape input into blocks and sum across block dimension
result = result + torch.sum(
    x.reshape(*x.shape[:-1], x.size(-1) // r, r),
    dim=-2
) @ bone
```

#### BAT Method (init_weights="bat")
Block-wise affine transform with matrix multiplication:
```python
# weight_bone shape: [out_features//r, r, r]
# Applies block-diagonal transformation with coupling
w = (orig_weight.reshape(...).permute(...) @ weight_bone + weight_bone)
output_tensor = w.permute(...).reshape(orig_weight.shape)
```

### Initialization Strategies

**BONE Initialization** (`reset_bone_parameters`):
```python
# Initialize to zeros (identity at start)
self.bone_block = nn.Parameter(torch.zeros(r, out_features))
```

**BAT Initialization** (`reset_bat_parameters`):
```python
# Requires full divisibility: in_features % r == 0 and out_features % r == 0
self.bone_block = nn.Parameter(torch.zeros(out_features // r, r, r))
```

**Random Initialization** (`reset_bone_parameters_random`):
```python
# Kaiming uniform initialization
nn.init.kaiming_uniform_(self.bone_block, a=math.sqrt(5))
```

### Merge/Unmerge Operations

**Merge Process**:
1. Compute delta weight using active adapter's bone_block
2. For BAT: Add delta to original weight
3. For BONE: Replace weight with transformed version
4. Append adapter to `merged_adapters` list

**Unmerge Process**:
1. For BAT: Compute inverse transformation using `torch.inverse(I + bone_block)`
2. For BONE: Subtract bone_block transformations
3. Restore original weight values

### Forward Pass Logic

**Adapter Disabled**: Pass through base layer
**Adapters Merged**: Use modified base layer weights
**Adapters Active**:
- **BAT mode**: Recompute weight transformation and apply via `F.linear`
- **BONE mode**: Apply block-wise transformation to activations

## I/O Contract

### Input
- **x**: `torch.Tensor` - Input tensor with shape `[batch, ..., in_features]`
- **args/kwargs**: Additional arguments passed to base layer

### Output
- **result**: `torch.Tensor` - Transformed output with shape `[batch, ..., out_features]`
- Maintains input dtype through transformations

### Constraints
- **BAT mode**: Requires `in_features % r == 0` and `out_features % r == 0`
- **BONE mode**: Handles arbitrary dimensions with padding if needed
- Supports safe merge with NaN detection
- CPU float16/bfloat16 operations cast to float32 for stability

## Usage Examples

### Basic BONE Adapter
```python
from peft import BoneConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure BONE
config = BoneConfig(
    r=64,
    target_modules=["q_proj", "v_proj"],
    init_weights=True,  # Use BONE method
    bias="none"
)

# Apply BONE adapter
peft_model = get_peft_model(model, config)
print(f"Trainable parameters: {peft_model.num_parameters(only_trainable=True)}")
```

### BAT Variant
```python
# Configure BAT (Block-wise Affine Transform)
config = BoneConfig(
    r=64,  # Must divide evenly into layer dimensions
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    init_weights="bat",  # Use BAT method
)

peft_model = get_peft_model(model, config)
```

### Diffusion Model Application
```python
from diffusers import StableDiffusionPipeline
from peft import BoneModel, BoneConfig

# Configure for text encoder
config_te = BoneConfig(
    r=8,
    target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
    init_weights=True,
)

# Configure for U-Net
config_unet = BoneConfig(
    r=8,
    target_modules=[
        "proj_in", "proj_out", "to_k", "to_q", "to_v",
        "to_out.0", "ff.net.0.proj", "ff.net.2",
    ],
    init_weights=True,
)

# Load and adapt pipeline
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline.text_encoder = BoneModel(pipeline.text_encoder, config_te, "default")
pipeline.unet = BoneModel(pipeline.unet, config_unet, "default")
```

### Merge and Save
```python
# Merge adapters into base weights
peft_model.merge_adapter()

# Save merged model
peft_model.save_pretrained("./bone_merged_model")

# Unmerge if needed
peft_model.unmerge_adapter()
```

## Related Pages
- [[LoRA Tuner Implementation]] - Similar low-rank adaptation approach
- [[MISS Tuner Implementation]] - Replacement for BONE in v0.19.0+
- [[Adapter Merge Strategies]] - Techniques for combining multiple adapters
- [[PEFT Configuration Guide]] - General PEFT configuration patterns
- [[Parameter-Efficient Fine-Tuning Methods]] - Overview of PEFT techniques
