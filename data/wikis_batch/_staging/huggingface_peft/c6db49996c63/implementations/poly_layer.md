# Poly Layer Implementation

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/poly/layer.py`
- **Lines**: 165
- **Purpose**: Poly (Polytropon) multi-skill LoRA layer with routing

## Overview

Poly implements multi-skill parameter-efficient adaptation where multiple LoRA-like adapters (called "skills") are combined using learned routing weights. Each layer contains n_skills different low-rank adaptations, and a router determines how to mix them based on task IDs or input features.

## Key Components

### PolyLayer (Base Class)

**Attributes**:
- `adapter_layer_names`: `("poly_lora_A", "poly_lora_B", "poly_router")` - Trainable components
- `other_param_names`: `("r", "n_tasks", "n_skills", "n_splits")` - Configuration
- `r`: Dict of ranks per adapter
- `n_tasks`: Dict of number of tasks
- `n_skills`: Dict of number of skills (LoRA modules)
- `n_splits`: Dict of splits (for Multi-Head Routing)
- `poly_lora_A`: ParameterDict storing skill A matrices (n_splits × n_skills × d_in × rank)
- `poly_lora_B`: ParameterDict storing skill B matrices (n_splits × n_skills × rank × d_out)
- `poly_router`: ModuleDict storing routing modules

**Key Methods**:

1. **`__init__(base_layer, **kwargs)`**
   - Only supports nn.Linear layers
   - Extracts in_features and out_features

2. **`update_layer(adapter_name, poly_config, inference_mode=False)`**
   - **Validation**: r > 0
   - **Configuration Storage**:
     - r, n_tasks, n_skills, n_splits, poly_type
   - **Parameter Creation**:
     - `poly_lora_A`: (n_splits, n_skills, in_features // n_splits, r)
     - `poly_lora_B`: (n_splits, n_skills, r, out_features // n_splits)
   - **Router Creation**: Calls `get_router(poly_config)`
   - **Initialization**: Calls `reset_poly_parameters`

3. **`reset_poly_parameters(adapter_name, init_weights)`**
   - **A Matrices**: Kaiming uniform initialization (transposed)
   - **B Matrices**:
     - If `init_weights=True`: Zeros (LoRA-style)
     - If `init_weights=False`: Kaiming uniform
   - **Router**: Calls `router.reset()`

### Linear (Implementation Class)

**Constructor Parameters**:
- `base_layer`: Original Linear layer
- `adapter_name`: Adapter name
- `poly_config`: PolyConfig instance

**Forward Method**:

```python
def forward(self, x, *args, task_ids=None, **kwargs):
    previous_dtype = x.dtype

    if self.disable_adapters:
        result = self.base_layer(x)
    else:
        result = self.base_layer(x)
        for active_adapter in self.active_adapters:
            if active_adapter not in self.poly_lora_A.keys():
                continue

            r = self.r[active_adapter]
            poly_router = self.poly_router[active_adapter]
            poly_lora_A = self.poly_lora_A[active_adapter]
            poly_lora_B = self.poly_lora_B[active_adapter]

            # Get mixing weights from router
            # Shape: (batch_size, n_splits, n_skills)
            mixing_weights = poly_router(task_ids=task_ids, input_ids=x)
            bs, n_splits, n_skills = mixing_weights.size()

            # Combine skills using mixing weights
            # A: (n_splits, n_skills, D // n_splits, rank)
            # -> (bs, n_splits, D // n_splits, rank)
            A = torch.einsum("bqs,qsdr->bqdr", (mixing_weights, poly_lora_A))

            # B: (n_splits, n_skills, rank, D // n_splits)
            # -> (bs, n_splits, rank, D // n_splits)
            B = torch.einsum("bqs,qsrd->bqrd", (mixing_weights, poly_lora_B))

            # Reshape for batch matrix multiplication
            A = A.reshape(bs, in_features, r)
            B = B.transpose(1, 2).reshape(bs, r, out_features)

            # Apply LoRA-style adaptation: x @ A @ B
            x = x.to(A.dtype)
            result += x.bmm(A).bmm(B) / r

    result = result.to(previous_dtype)
    return result
```

## Mathematical Formulation

Poly combines multiple LoRA-style adaptations:

```
For each split s and skill k:
    A[s,k]: (d_in / n_splits) × rank
    B[s,k]: rank × (d_out / n_splits)

Mixing weights w[s,k] = router(task_id, x)

Combined adaptation:
    A_combined = Σ_k w[s,k] * A[s,k]  for each split s
    B_combined = Σ_k w[s,k] * B[s,k]  for each split s

Output = base_layer(x) + x @ A_combined @ B_combined / rank
```

### Einsum Operations

```python
# Combine A matrices
# mixing_weights: (batch, n_splits, n_skills)
# poly_lora_A: (n_splits, n_skills, d_in/n_splits, rank)
# Result: (batch, n_splits, d_in/n_splits, rank)
A = torch.einsum("bqs,qsdr->bqdr", (mixing_weights, poly_lora_A))

# Combine B matrices
# poly_lora_B: (n_splits, n_skills, rank, d_out/n_splits)
# Result: (batch, n_splits, rank, d_out/n_splits)
B = torch.einsum("bqs,qsrd->bqrd", (mixing_weights, poly_lora_B))
```

## Multi-Head Routing (MHR)

When `n_splits > 1`, enables Multi-Head Routing:

### Splitting Strategy

**Input dimension**: d_in = n_splits × (d_in / n_splits)
**Output dimension**: d_out = n_splits × (d_out / n_splits)

**Example** (d_in=768, n_splits=3):
- Split 0: handles dimensions 0-255
- Split 1: handles dimensions 256-511
- Split 2: handles dimensions 512-767

### Independent Routing

Each split can route differently:
```python
# Split 0 might use: 70% skill_1 + 30% skill_3
# Split 1 might use: 90% skill_2 + 10% skill_4
# Split 2 might use: 50% skill_1 + 50% skill_2
```

This provides finer-grained control and higher capacity.

## Router Types

From config.py and router.py:

### 1. Poly Router (Default)

**Task-based routing**:
```python
class PolyRouter(nn.Module):
    def __init__(self, n_tasks, n_skills, n_splits):
        # Learned routing weights per task
        self.weights = nn.Parameter(torch.zeros(n_tasks, n_splits, n_skills))

    def forward(self, task_ids, input_ids):
        # task_ids: (batch_size,)
        # Select weights for each task in batch
        weights = self.weights[task_ids]  # (batch, n_splits, n_skills)
        # Softmax to get mixing probabilities
        return F.softmax(weights, dim=-1)

    def reset(self):
        # Initialize with uniform distribution
        nn.init.constant_(self.weights, 1.0 / self.n_skills)
```

**Properties**:
- One set of weights per task
- Task ID required at forward pass
- Deterministic given task ID

## Parameter Count

### Standard Poly

For layer d_in × d_out:
- **LoRA A per skill**: n_splits × (d_in / n_splits) × r
- **LoRA B per skill**: n_splits × r × (d_out / n_splits)
- **Total per skill**: r(d_in + d_out)
- **Total for n_skills**: n_skills × r(d_in + d_out)
- **Router**: n_tasks × n_splits × n_skills

**Example** (d_in=d_out=4096, r=8, n_skills=4, n_splits=1, n_tasks=10):
- Skills: 4 × 8 × (4096 + 4096) = 262,144
- Router: 10 × 1 × 4 = 40
- **Total**: 262,184

### With Multi-Head Routing

**Example** (same, but n_splits=4):
- Skills: Same (262,144)
- Router: 10 × 4 × 4 = 160
- **Total**: 262,304

MHR adds minimal overhead but provides finer control.

## Initialization

### A Matrices (Kaiming Uniform)
```python
for skill in range(n_skills):
    for split in range(n_splits):
        param = torch.empty((r, d))
        torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        # Store transposed
        poly_lora_A[split, skill, :, :] = param.T
```

### B Matrices
**If init_weights=True** (default):
```python
torch.nn.init.zeros_(poly_lora_B)  # LoRA initialization
```

**If init_weights=False**:
```python
# Kaiming uniform (for testing)
for skill in range(n_skills):
    for split in range(n_splits):
        param = torch.empty((d, r))
        torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        poly_lora_B[split, skill, :, :] = param.T
```

## Design Patterns

### 1. Task ID Injection
Requires task_ids at forward pass:
```python
output = model(input, task_ids=task_ids)
```

Handled via pre-hooks in PolyModel

### 2. Batch Matrix Multiplication
Efficient per-sample LoRA:
```python
x.bmm(A).bmm(B)  # Batch matrix multiply
```

Each sample in batch can have different routing

### 3. Einsum for Mixing
Elegant skill combination:
```python
A = torch.einsum("bqs,qsdr->bqdr", (mixing_weights, poly_lora_A))
```

### 4. Split-Based Architecture
Divides computation for parallelism and capacity

## Usage Example

```python
from peft.tuners.poly.layer import Linear
from peft import PolyConfig

config = PolyConfig(
    r=8,
    n_tasks=10,
    n_skills=4,
    n_splits=1,
    poly_type="poly"
)

poly_layer = Linear(
    base_layer=nn.Linear(768, 768),
    adapter_name="default",
    poly_config=config
)

# Forward pass with task IDs
task_ids = torch.tensor([0, 1, 2, 0])  # Batch of 4
output = poly_layer(input_tensor, task_ids=task_ids)
```

## Comparison with Standard LoRA

### LoRA
- Single low-rank adaptation
- Fixed for all inputs
- r(d_in + d_out) parameters

### Poly
- Multiple skills (n_skills adaptations)
- Dynamic mixing based on task/input
- n_skills × r(d_in + d_out) parameters for skills
- Plus routing parameters

### Benefits of Poly
- Multi-task learning
- Task-specific specialization
- Flexible skill composition
- Shared skill library

## Limitations

1. **Layer Support**: Only nn.Linear
2. **Task ID Requirement**: Must provide task_ids
3. **Memory**: n_skills × LoRA memory
4. **Complexity**: More complex than standard LoRA

## Integration Points

- Imports `BaseTunerLayer` from `peft.tuners.tuners_utils`
- Uses `get_router` from `.router`
- PyTorch einsum and batch operations

## References

- **Polytropon Paper**: https://huggingface.co/papers/2202.13914
- **MHR Paper**: https://huggingface.co/papers/2211.03831
- **Concept**: Multi-skill parameter-efficient adaptation with learned routing
