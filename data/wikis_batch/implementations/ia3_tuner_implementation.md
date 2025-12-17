# IA3 Tuner Implementation

## Metadata
- **Type**: PEFT Tuner Implementation
- **Module Path**: `src/peft/tuners/ia3/`
- **Components**: Layer, Config, Model
- **Lines of Code**: 757 (layer: 330, config: 112, model: 315)
- **PEFT Type**: `PeftType.IA3`
- **Paper**: https://huggingface.co/papers/2205.05638

## Overview

IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations) is a parameter-efficient fine-tuning method that learns to rescale activations using learned vectors. Unlike LoRA which adds matrices, IA3 multiplies activations by learned scaling vectors, achieving extreme parameter efficiency.

**Key Features**:
- Elementwise rescaling of activations or weights
- Separate treatment for attention (multiply outputs) vs. feedforward (multiply inputs)
- Extremely parameter-efficient: only 0.01% of parameters for T5-base
- Supports Linear, Conv2d, Conv3d, and Conv1D layers
- Compatible with 8-bit and 4-bit quantization

## Core Components

### 1. IA3Layer (`layer.py`)

Base adapter layer implementing activation rescaling.

**Key Classes**:
- `IA3Layer`: Base tuner layer with rescaling vector management
- `Linear`: IA3 implementation for Linear/Conv1D layers
- `Conv2d`: IA3 implementation for 2D convolutional layers
- `Conv3d`: IA3 implementation for 3D convolutional layers

**Adapter Parameters**:
```python
adapter_layer_names = ("ia3_l",)  # Learned scaling vectors
```

**State Management**:
- `ia3_l`: ParameterDict storing learned vectors
  - Feedforward: `[1, in_features]` - rescale inputs
  - Attention: `[out_features, 1]` - rescale outputs
- `is_feedforward`: Boolean flag determining rescaling mode
- `merged_adapters`: List of currently merged adapters

### 2. IA3Config (`config.py`)

Configuration dataclass for IA3 adapters.

**Key Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_modules` | Optional[Union[list[str], str]] | None | Module names or regex to apply IA3 |
| `exclude_modules` | Optional[Union[list[str], str]] | None | Module names or regex to exclude |
| `feedforward_modules` | Optional[Union[list[str], str]] | None | Modules treated as feedforward |
| `fan_in_fan_out` | bool | False | Set True for Conv1D layers (e.g., GPT-2) |
| `modules_to_save` | Optional[list[str]] | None | Additional modules to train/save |
| `init_ia3_weights` | bool | True | Initialize scaling vectors to ones |

**Validation**:
- `feedforward_modules` must be subset of `target_modules` (when both are sets)
- Converts parameters to sets if provided as lists
- Auto-detection of target/feedforward modules based on model architecture

### 3. IA3Model (`model.py`)

Model wrapper that injects IA3 adapters into pretrained models.

**Class Attributes**:
- `prefix`: "ia3_"
- `tuner_layer_cls`: `IA3Layer`
- `target_module_mapping`: `TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING`
- `feedforward_module_mapping`: `TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING`

**Special Features**:
- Weighted adapter combination support
- Automatic feedforward module detection
- 8-bit and 4-bit quantization support

## Implementation Details

### Rescaling Strategy

#### Feedforward Modules
Rescale **inputs** before weight multiplication:

```python
# Learned vector: [1, in_features]
ia3_scaling = self.ia3_l[adapter_name].flatten()

# Apply to input
x_scaled = x * ia3_scaling

# Then apply base layer
result = self.base_layer(x_scaled)
```

**Common feedforward modules**: `fc1`, `fc2`, `w0`, `output.dense`

#### Attention Modules
Rescale **outputs** after weight multiplication:

```python
# Learned vector: [out_features, 1]
ia3_scaling = self.ia3_l[adapter_name].flatten()

# Apply base layer first
result = self.base_layer(x)

# Then rescale output
result = result * ia3_scaling
```

**Common attention modules**: `q`, `k`, `v`, `q_proj`, `k_proj`, `v_proj`

### Initialization

**Default Initialization** (`init_ia3_weights=True`):
```python
# Initialize to ones (identity at start)
nn.init.constant_(self.ia3_l[adapter_name], 1.0)
```

**Benefits**:
- Model behavior unchanged at initialization
- Gradual learning of rescaling factors
- Stable training from start

### Merge/Unmerge Operations

**Merge Process** (Linear layers):
```python
# Transpose if needed
ia3_l = transpose(self.ia3_l[adapter_name].data, self.fan_in_fan_out)

# Elementwise multiplication with base weights
base_layer.weight.data = base_layer.weight.data * ia3_l

# For attention modules, rescale bias too
if not is_feedforward and base_layer.bias is not None:
    scaling = self.ia3_l[adapter_name].reshape(base_layer.bias.shape)
    base_layer.bias.data = base_layer.bias.data * scaling.data
```

**Unmerge Process**:
```python
# Inaccurate due to division (adds small epsilon for stability)
ia3_l = self.ia3_l[adapter_name].data + 1e-8
base_layer.weight.data = base_layer.weight.data / ia3_l

# Warning: "Unmerge result can be inaccurate for (IA)^3"
```

**Note**: Unlike LoRA/HRA, unmerging is approximate due to division.

### Convolutional Layers

**Conv2d/Conv3d rescaling**:
```python
# Feedforward: rescale input channels
if is_feedforward:
    weights_size = (1, in_channels) + (1,) * (kernel_dim - 2)
    ia3_scaling = self.ia3_l[adapter_name]  # [1, in_channels, 1, 1]
    x_scaled = x * ia3_scaling
    result = self.base_layer(x_scaled)

# Attention: rescale output channels
else:
    result = self.base_layer(x)
    ia3_scaling = self.ia3_l[adapter_name]  # [1, out_channels, 1, 1]
    result = result * ia3_scaling
```

### Weighted Adapter Combination

IA3Model supports combining multiple adapters with weights:

```python
def add_weighted_adapter(
    self,
    adapters: list[str],
    weights: list[float],
    adapter_name: str
):
    # Validate adapters exist and are compatible
    new_target_modules, new_feedforward_modules = (
        self._check_add_weighted_adapter(adapters)
    )

    # Create new adapter config
    self.peft_config[adapter_name] = replace(
        self.peft_config[adapters[0]],
        target_modules=new_target_modules,
        feedforward_modules=new_feedforward_modules,
    )

    # Inject adapter layers
    self.inject_adapter(self.model, adapter_name)

    # Compute weighted combination
    for key in model.named_modules():
        if isinstance(target, IA3Layer):
            target_ia3_l = target.ia3_l[adapter_name]
            target_ia3_l.data = target_ia3_l.data.zero_()

            for adapter, weight in zip(adapters, weights):
                current_ia3_l = target.ia3_l[adapter]
                target_ia3_l.data += current_ia3_l.data * weight
```

## I/O Contract

### Input
- **x**: `torch.Tensor` - Input tensor
  - Linear: `[batch, ..., in_features]`
  - Conv2d: `[batch, in_channels, height, width]`
  - Conv3d: `[batch, in_channels, depth, height, width]`
- **args/kwargs**: Additional arguments passed to base layer

### Output
- **result**: `torch.Tensor` - Rescaled output maintaining input shape structure
- Maintains input dtype through transformations

### Constraints
- Learned vectors initialized to ones (identity)
- Feedforward vs. attention determined by `feedforward_modules` config
- Supports safe merge with NaN detection
- Unmerge is approximate (not exact like LoRA/HRA)
- Cannot merge when model is loaded in 8-bit or 4-bit mode

## Usage Examples

### Basic IA3 Adapter (Seq2Seq)
```python
from peft import IA3Config, get_peft_model
from transformers import AutoModelForSeq2SeqLM

# Load T5 model
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

# Configure IA3
config = IA3Config(
    task_type="SEQ_2_SEQ_LM",
    target_modules=["k", "v", "w0"],  # k, v = attention; w0 = feedforward
    feedforward_modules=["w0"],        # Specify feedforward modules
    init_ia3_weights=True
)

# Apply IA3 adapter
peft_model = get_peft_model(model, config)
print(f"Trainable parameters: {peft_model.num_parameters(only_trainable=True)}")
# T5-base: ~220M parameters -> ~40K trainable (0.02%)
```

### Causal Language Modeling
```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")

# Configure for GPT-2 architecture
config = IA3Config(
    task_type="CAUSAL_LM",
    target_modules=["c_attn", "mlp.c_proj"],  # Attention and feedforward
    feedforward_modules=["mlp.c_proj"],       # Feedforward paths
    fan_in_fan_out=True,  # Important for Conv1D in GPT-2!
    init_ia3_weights=True
)

peft_model = get_peft_model(model, config)
```

### Auto-Detection of Modules
```python
# Let PEFT auto-detect target and feedforward modules
config = IA3Config(
    task_type="SEQ_2_SEQ_LM",
    # target_modules and feedforward_modules will be auto-detected
    init_ia3_weights=True
)

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
peft_model = get_peft_model(model, config)
```

### Manual Module Specification
```python
# Manually specify all modules
config = IA3Config(
    target_modules=["q_proj", "k_proj", "v_proj", "fc1", "fc2"],
    feedforward_modules=["fc1", "fc2"],
    exclude_modules=["lm_head"],  # Exclude output layer
    init_ia3_weights=True
)

peft_model = get_peft_model(model, config)
```

### Regex-Based Module Selection
```python
# Use regex to match modules
config = IA3Config(
    target_modules=r".*\.(q|k|v)_proj$",  # All q/k/v projection layers
    feedforward_modules=r".*\.fc[12]$",    # All fc1 and fc2 layers
    init_ia3_weights=True
)

peft_model = get_peft_model(model, config)
```

### 8-bit Quantization Support
```python
# Load model in 8-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
    device_map="auto"
)

# IA3 works with quantized models
config = IA3Config(
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    feedforward_modules=[],
    init_ia3_weights=True
)

peft_model = get_peft_model(model, config)
# Note: Cannot merge adapters when model is quantized
```

### Weighted Adapter Combination
```python
# Train multiple task-specific adapters
model.load_adapter("adapter_task1", adapter_name="task1")
model.load_adapter("adapter_task2", adapter_name="task2")

# Combine adapters with weights
model.add_weighted_adapter(
    adapters=["task1", "task2"],
    weights=[0.7, 0.3],
    adapter_name="combined"
)

# Use combined adapter
model.set_adapter("combined")
```

### Training and Inference
```python
# Train with IA3
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()

# Merge for inference (if not quantized)
peft_model.merge_adapter()
peft_model.save_pretrained("./ia3_merged_model")

# Or save adapter weights only
peft_model.save_pretrained("./ia3_adapter", safe_serialization=True)
```

## Performance Characteristics

### Parameter Efficiency
- **Feedforward module**: `in_features` parameters
- **Attention module**: `out_features` parameters
- **Total (e.g., T5-base)**: ~40K parameters (0.02% of base model)
- **Compared to LoRA**: 10-100× fewer parameters

### Memory Usage
- Minimal memory overhead
- Only stores scaling vectors
- Compatible with quantization (8-bit/4-bit)

### Computational Complexity
- **Forward pass**: O(features) - just elementwise multiplication
- **Backward pass**: Extremely efficient gradient computation
- **Negligible overhead** compared to base model

### Training Characteristics
- Very fast training (fewer parameters to update)
- Good performance on many tasks
- May underperform LoRA on complex tasks requiring more expressiveness
- Excellent for low-resource scenarios

## Comparison with Other Methods

| Method | Parameters | Expressiveness | Merge | Quantization |
|--------|------------|----------------|-------|--------------|
| IA3 | 0.01-0.1% | Lower | Approximate | ✓ |
| LoRA | 0.1-1% | Higher | Exact | ✓ |
| Adapter | 1-5% | High | N/A | ✓ |
| Prefix Tuning | 0.1-1% | Medium | N/A | ✓ |

## Related Pages
- [[LoRA Tuner Implementation]] - Low-rank adaptation (more parameters)
- [[Adapter Tuner Implementation]] - Bottleneck adapter layers
- [[Prefix Tuning Implementation]] - Prefix-based fine-tuning
- [[Quantization with PEFT]] - Using PEFT with quantized models
- [[PEFT Configuration Guide]] - General PEFT configuration patterns
- [[Parameter-Efficient Fine-Tuning Comparison]] - Comparison of PEFT methods
