# GraLoRA Tuner Implementation

## Metadata
- **Type**: PEFT Tuner Implementation
- **Module Path**: `src/peft/tuners/gralora/`
- **Components**: Layer, Config, Model
- **Lines of Code**: 716 (layer: 392, config: 182, model: 142)
- **PEFT Type**: `PeftType.GRALORA`

## Overview

GraLoRA (Gradient Low-Rank Adaptation) implements block-wise low-rank adaptation with information exchange between blocks. Unlike standard LoRA, GraLoRA divides weight matrices into subblocks and applies low-rank updates with cross-block communication, improving expressiveness while maintaining the same parameter count.

**Key Features**:
- Block-wise decomposition with configurable subblock count (`gralora_k`)
- Information exchange between blocks via tensor reshaping
- Optional hybrid mode combining GraLoRA + vanilla LoRA
- Same parameter count as LoRA with rank `r`, but expressivity multiplied by `gralora_k`

## Core Components

### 1. GraloraLayer (`layer.py`)

Base adapter layer implementing block-wise low-rank adaptations.

**Key Classes**:
- `GraloraLayer`: Base tuner layer with block-wise adapter management
- `Linear`: GraLoRA implementation for Linear/Conv1D layers

**Adapter Parameters**:
```python
adapter_layer_names = ("gralora_A", "gralora_B", "gralora_A_general", "gralora_B_general")
other_param_names = ("r", "hybrid_r", "alpha", "scaling", "gralora_dropout")
```

**State Management**:
- `gralora_A`: ParameterDict storing A matrices `[N, in_features//N, rank]`
- `gralora_B`: ParameterDict storing B matrices `[N, rank, out_features//N]`
- `gralora_A_general`: ModuleDict for hybrid LoRA A (Linear layers)
- `gralora_B_general`: ModuleDict for hybrid LoRA B (Linear layers)
- `r`, `alpha`, `gralora_k`, `hybrid_r`, `scaling`: Per-adapter hyperparameters

### 2. GraloraConfig (`config.py`)

Configuration dataclass for GraLoRA adapters.

**Key Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `r` | int | 32 | GraLoRA rank (total across all blocks) |
| `hybrid_r` | int | 0 | Rank for vanilla LoRA component (if > 0) |
| `alpha` | int | 64 | Scaling factor (scale = alpha / (r + hybrid_r)) |
| `gralora_dropout` | float | 0.0 | Dropout probability for GraLoRA adapters |
| `gralora_k` | int | 2 | Number of subblocks (r must be divisible by k) |
| `target_modules` | Optional[Union[list[str], str]] | None | Module names or regex to apply GraLoRA |
| `fan_in_fan_out` | bool | False | Set True for Conv1D layers (e.g., GPT-2) |
| `bias` | str | "none" | Bias handling: 'none', 'all', 'gralora_only' |
| `init_weights` | bool | True | Initialize A randomly, B to zeros |
| `layers_to_transform` | Optional[Union[list[int], int]] | None | Specific layer indices to transform |
| `layers_pattern` | Optional[str] | None | Layer pattern for selective transformation |
| `modules_to_save` | Optional[list[str]] | None | Additional modules to train/save |

**Validation**:
- `r % gralora_k == 0` (rank must be divisible by number of blocks)
- `in_features % gralora_k == 0` and `out_features % gralora_k == 0`
- Converts `target_modules` to set if provided as list

**Recommendations**:
- `gralora_k=2` for rank ≤ 32
- `gralora_k=4` for rank ≥ 64

### 3. GraloraModel (`model.py`)

Model wrapper that injects GraLoRA adapters into pretrained models.

**Class Attributes**:
- `prefix`: "gralora_"
- `tuner_layer_cls`: `GraloraLayer`
- `target_module_mapping`: `TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING`

## Implementation Details

### Block-Wise Decomposition

GraLoRA divides the weight matrix into `gralora_k` subblocks and applies low-rank updates with information exchange:

```python
# Parameters structure
gralora_A: [N, in_features//N, rank]       # N = gralora_k
gralora_B: [N, rank, out_features//N]
subblock_rank = rank // N

# Forward computation with cross-block information exchange
# 1. Apply local subblock transformations
output = einsum('blni, nir -> blnr', x.view(B, L, N, in_features//N), gralora_A)

# 2. Reshape to enable cross-block communication
output = output.view(B, L, N, N, subblock_rank).permute(0, 1, 3, 2, 4)

# 3. Flatten and apply output transformation
output = einsum('bljr, jro -> bljo', output.reshape(B, L, N, rank), gralora_B)

# 4. Reshape to final output
result = output.reshape(B, L, -1)
```

### Hybrid GraLoRA Mode

When `hybrid_r > 0`, combines GraLoRA with standard LoRA:

```python
# GraLoRA component
gralora_output = <block-wise computation>

# Vanilla LoRA component
hybrid_output = gralora_B_general(gralora_A_general(x))

# Combined scaling
scaling = alpha / (r + hybrid_r)
result = base_output + scaling * (gralora_output + hybrid_output)
```

### Weight Delta Computation

For merging adapters into base weights:

```python
def get_delta_weight(self, adapter):
    # Scatter A matrix to enable cross-block access
    gralora_A_scattered = torch.zeros(in_features, gralora_k, rank)
    gralora_A_scattered.scatter_(..., gralora_A[n_indices, i_indices, :])

    # Compute block-wise delta with reshaping for information exchange
    delta_weight = einsum(
        'ikr, kro -> iko',
        gralora_A_scattered.view(...).permute(...).reshape(...),
        gralora_B
    ).reshape(in_features, out_features).T

    # Add hybrid component if present
    if hybrid_r > 0:
        delta_weight += gralora_B_general.weight @ gralora_A_general.weight

    return delta_weight * scaling
```

### Initialization Strategy

**Default Initialization** (`init_weights=True`):
```python
# A matrices: Random (Kaiming uniform)
for each block:
    nn.init.kaiming_uniform_(gralora_A[i], a=math.sqrt(5))
    nn.init.zeros_(gralora_B[i])  # Identity at initialization

# Hybrid components (if hybrid_r > 0)
nn.init.kaiming_uniform_(gralora_A_general.weight, a=math.sqrt(5))
nn.init.zeros_(gralora_B_general.weight)
```

### 2D/3D Input Handling

The implementation handles both 2D and 3D inputs:

```python
# Handle 2D input: [batch, features] -> [batch, 1, features]
x_is_2d = x.ndim == 2
if x_is_2d:
    x = x.unsqueeze(1)

# Process with 3D operations
output = <gralora_computation>

# Squeeze back if input was 2D
if x_is_2d:
    output = output.squeeze(1)
```

## I/O Contract

### Input
- **x**: `torch.Tensor` - Input tensor with shape:
  - 2D: `[batch, in_features]`
  - 3D: `[batch, seq_len, in_features]`
- **args/kwargs**: Additional arguments passed to base layer

### Output
- **result**: `torch.Tensor` - Transformed output with same rank as input
- Maintains input dtype through transformations

### Constraints
- `in_features % gralora_k == 0` (enforced at layer creation)
- `out_features % gralora_k == 0` (enforced at layer creation)
- `r % gralora_k == 0` (enforced in config validation)
- Supports safe merge with NaN detection
- CPU float16/bfloat16 operations cast to float32 for stability

## Usage Examples

### Basic GraLoRA Adapter
```python
from peft import GraloraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

# Configure GraLoRA
config = GraloraConfig(
    r=32,           # Total rank
    alpha=64,       # Scaling alpha
    gralora_k=2,    # 2 subblocks (recommended for rank 32)
    target_modules=["q_proj", "v_proj"],
    gralora_dropout=0.0,
    init_weights=True
)

# Apply GraLoRA adapter
peft_model = get_peft_model(model, config)
print(f"Trainable parameters: {peft_model.num_parameters(only_trainable=True)}")
```

### Hybrid GraLoRA
```python
# Combine GraLoRA with vanilla LoRA
config = GraloraConfig(
    r=32,           # GraLoRA rank
    hybrid_r=16,    # Vanilla LoRA rank
    alpha=96,       # Scale = 96 / (32 + 16) = 2.0
    gralora_k=2,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    gralora_dropout=0.1
)

peft_model = get_peft_model(model, config)
```

### High-Rank Configuration
```python
# Use gralora_k=4 for higher ranks
config = GraloraConfig(
    r=64,           # Higher rank
    alpha=128,
    gralora_k=4,    # 4 subblocks (recommended for rank ≥ 64)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    init_weights=True
)

peft_model = get_peft_model(model, config)
```

### Conv1D Layers (GPT-2)
```python
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("gpt2")

# GPT-2 uses Conv1D, so set fan_in_fan_out=True
config = GraloraConfig(
    r=32,
    alpha=64,
    gralora_k=2,
    target_modules=["c_attn", "c_proj"],
    fan_in_fan_out=True,  # Important for Conv1D!
    init_weights=True
)

peft_model = get_peft_model(model, config)
```

### Selective Layer Application
```python
# Apply only to specific transformer layers
config = GraloraConfig(
    r=32,
    alpha=64,
    gralora_k=2,
    target_modules=["q_proj", "v_proj"],
    layers_to_transform=[0, 1, 2],  # Only first 3 layers
    layers_pattern="model.layers"   # Pattern to match
)

peft_model = get_peft_model(model, config)
```

### Training and Merging
```python
# Train the model
trainer.train()

# Merge adapters for inference
peft_model.merge_adapter()

# Save merged model
peft_model.save_pretrained("./gralora_merged_model")

# Or save adapter weights only
peft_model.save_pretrained("./gralora_adapter", safe_serialization=True)
```

## Performance Characteristics

### Parameter Efficiency
- **GraLoRA only**: Same as LoRA with rank `r`
- **Hybrid GraLoRA**: Parameters = LoRA(r) + LoRA(hybrid_r)
- **Expressivity**: Multiplied by `gralora_k` compared to standard LoRA

### Memory Usage
- Forward pass requires intermediate tensor reshaping
- Memory overhead: O(batch_size × seq_len × rank × gralora_k)
- Dropout requires additional memory if `gralora_dropout > 0`

### Computational Complexity
- Similar to LoRA with same total rank
- Additional cost from tensor permutation operations
- Hybrid mode adds standard LoRA computation overhead

## Related Pages
- [[LoRA Tuner Implementation]] - Standard low-rank adaptation
- [[VeRA Tuner Implementation]] - Vector-based random matrix adaptation
- [[DoRA Tuner Implementation]] - Weight-decomposed low-rank adaptation
- [[Adapter Merge Strategies]] - Techniques for combining multiple adapters
- [[PEFT Configuration Guide]] - General PEFT configuration patterns
