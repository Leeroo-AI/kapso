# MissLayer (Householder Reflection Adaptation)

## Implementation Overview

**File:** `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/miss/layer.py`
**Lines of Code:** 393
**Language:** Python

MissLayer implements the MiSS (Minimally Structured Sparse) adapter using Householder reflections for parameter-efficient fine-tuning. This method applies low-rank decomposition along input features with three variant modes: standard MiSS, BAT (Block-wise Adaptive Transformations), and Mini mode for maximum parameter efficiency.

## Core Components

### 1. Base Layer Class

**Class:** `MissLayer(BaseTunerLayer)`

The foundation class that manages MISS adapters with Householder reflection matrices:

```python
class MissLayer(BaseTunerLayer):
    adapter_layer_names = ("miss_block",)
    other_param_names = ("miss_r", "miss_dropout", "miss_mini_r")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.miss_r = {}
        self.miss_dropout = nn.ModuleDict({})
        self.miss_mini_r = {}
        self.miss_block = nn.ParameterDict({})
        self._disable_adapters = False
        self.merged_adapters = []
```

**Key Attributes:**
- `miss_block`: Stores trainable Householder reflection parameters
- `miss_r`: Rank for in_features dimension decomposition
- `miss_mini_r`: Rank for out_features dimension (Mini mode)
- `miss_dropout`: Dropout layers per adapter

### 2. Linear Layer Implementation

**Class:** `MissLinear(nn.Module, MissLayer)`

Implements MiSS for dense linear layers with three operational variants:

```python
class MissLinear(nn.Module, MissLayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        mini_r: int = 0,
        miss_dropout: float = 0.0,
        init_weights: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        MissLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, mini_r, miss_dropout, init_weights, **kwargs)
        self.miss_fn = init_weights
```

## Key Methods

### Layer Configuration

**Method:** `update_layer()`

Configures adapter parameters and initializes weights based on variant:

```python
def update_layer(
    self,
    adapter_name: str,
    r: int,
    mini_r: int,
    miss_dropout,
    init_weights: bool | str,
    inference_mode: bool = False,
    **kwargs,
) -> None:
    if r <= 0:
        raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

    self.miss_r[adapter_name] = r
    self.miss_mini_r[adapter_name] = mini_r

    # Create dropout layer
    if miss_dropout > 0.0:
        miss_dropout_layer = nn.Dropout(p=miss_dropout)
    else:
        miss_dropout_layer = nn.Identity()

    self.miss_dropout[adapter_name] = miss_dropout_layer

    # Determine shape based on initialization variant
    if init_weights == "bat":
        if self.in_features % r != 0 or self.out_features % r != 0:
            raise ValueError("The weight matrix must be fully divisible into [r, r] blocks.")
        self.reset_bat_parameters(adapter_name, r)
    elif init_weights == "mini":
        if self.out_features % mini_r != 0:
            raise ValueError("out_features must be divisible by mini_r")
        self.reset_mini_parameters(adapter_name, r, mini_r)
    else:
        self.reset_miss_parameters(adapter_name, r)
```

**Variants:**
- **Standard (init_weights=True)**: Shape `[r, out_features]`, most general method
- **BAT (init_weights="bat")**: Shape `[out_features // r, r, r]`, enables nonlinear updates
- **Mini (init_weights="mini")**: Shape `[r, mini_r]`, smallest parameter count

### Forward Pass

**Method:** `forward()`

Applies MiSS transformation using Householder reflections:

```python
def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    previous_dtype = x.dtype

    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        if self.miss_fn == "bat":
            # BAT mode: Apply block-wise transformation to weights
            orig_weight = self.base_layer.weight.data.clone()
            for active_adapter in self.active_adapters:
                if active_adapter not in self.miss_block.keys():
                    continue
                delta_weight = self.get_delta_weight(active_adapter, orig_weight)
                orig_weight = orig_weight + delta_weight

            x = self._cast_input_dtype(x, orig_weight.dtype)
            bias = self._cast_input_dtype(self.base_layer.bias, orig_weight.dtype)
            result = F.linear(input=x, weight=orig_weight, bias=bias)
        else:
            # Standard/Mini mode: Apply element-wise transformation
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.miss_block.keys():
                    continue
                miss = self.miss_block[active_adapter]
                if self.miss_fn == "mini":
                    miss = miss.repeat(1, self.base_layer.out_features // self.miss_mini_r[active_adapter])

                dropout = self.miss_dropout[active_adapter]
                r = miss.size(0)
                # Handle non-divisible input dimensions
                if x.size(-1) % r != 0:
                    padding_size = (r - x.size(-1) % r) % r
                    x = F.pad(x, (0, padding_size))
                x = self._cast_input_dtype(x, miss.dtype)
                # Apply Householder reflection: sum over blocks then matrix multiply
                result = result + torch.sum(dropout(x).reshape(*x.shape[:-1], x.size(-1) // r, r), dim=-2) @ miss

    result = result.to(previous_dtype)
    return result
```

**Flow:**
1. Handle disabled/merged states
2. For BAT mode: Transform weights directly
3. For Standard/Mini: Transform activations
4. Apply dropout and Householder reflections
5. Handle input padding for non-divisible dimensions

### Weight Delta Computation (BAT Mode)

**Method:** `get_delta_weight()`

Computes block-diagonal weight updates for BAT variant:

```python
def get_delta_weight(self, adapter, orig_weight, re: bool = False) -> torch.Tensor:
    device = self.miss_block[adapter].device
    dtype = self.miss_block[adapter].dtype
    cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

    weight_miss = self.miss_block[adapter]
    if cast_to_fp32:
        weight_miss = weight_miss.float()
    orig_weight = orig_weight.to(weight_miss.dtype)

    r = weight_miss.size(-1)
    if re:
        # Unmerge: inverse transformation
        o = orig_weight.reshape(orig_weight.size(0) // r, r, orig_weight.size(1) // r, r).permute(2, 0, 1, 3)
        one = torch.eye(weight_miss.size(-1)).to(weight_miss.device)
        inv_I_plus_b = torch.inverse(one + weight_miss)
        inv_I_plus_b = inv_I_plus_b.to(weight_miss.dtype)
        w = (o - weight_miss) @ inv_I_plus_b
        output_tensor = w.permute(1, 2, 0, 3).reshape(*orig_weight.shape)
    else:
        # Merge: apply block-wise transformation
        w = (
            orig_weight.reshape(orig_weight.size(0) // r, r, orig_weight.size(1) // r, r).permute(2, 0, 1, 3)
            @ weight_miss
            + weight_miss
        )
        output_tensor = w.permute(1, 2, 0, 3).reshape(*orig_weight.shape)

    if cast_to_fp32:
        output_tensor = output_tensor.to(dtype=dtype)
        self.miss_block[adapter].data = weight_miss.to(dtype)

    return output_tensor
```

**Formula:** `W' = (W_block @ B + B)` where B is the Householder block

### Weight Delta Computation (Standard/Mini Mode)

**Method:** `get_delta_weight_miss()`

Computes element-wise weight updates:

```python
def get_delta_weight_miss(self, adapter, orig_weight, re: bool = False) -> torch.Tensor:
    device = self.miss_block[adapter].device
    dtype = self.miss_block[adapter].dtype
    cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

    weight_miss = self.miss_block[adapter]
    if cast_to_fp32:
        weight_miss = weight_miss.float()

    in_features = orig_weight.size(-1)
    out_features = orig_weight.size(0)
    r = weight_miss.size(0)

    # Mini mode: repeat weights along out_features
    if self.miss_fn == "mini":
        weight_miss = weight_miss.repeat(1, out_features // self.miss_mini_r[adapter])

    if in_features % r != 0:
        # Handle non-divisible dimensions
        last_size = in_features % r
        n_block = in_features // r
        n_block_size = n_block * r

        if re:
            # Subtract for unmerge
            orig_weight[:, :n_block_size] = (
                (orig_weight[:, :n_block_size].reshape(-1, n_block, r).permute(1, 2, 0) - weight_miss)
                .permute(2, 0, 1)
                .reshape(*orig_weight[:, :n_block_size].shape)
            )
            orig_weight[:, n_block_size:] = (
                orig_weight[:, n_block_size:] - (weight_miss.transpose(0, 1))[:, :last_size]
            )
        else:
            # Add for merge
            orig_weight[:, :n_block_size] = (
                (orig_weight[:, :n_block_size].reshape(-1, n_block, r).permute(1, 2, 0) + weight_miss)
                .permute(2, 0, 1)
                .reshape(*orig_weight[:, :n_block_size].shape)
            )
            orig_weight[:, n_block_size:] = (
                orig_weight[:, n_block_size:] + (weight_miss.transpose(0, 1))[:, :last_size]
            )
        output_tensor = orig_weight
    else:
        # Perfectly divisible case
        if re:
            w = orig_weight.reshape(-1, orig_weight.size(1) // r, r).permute(1, 2, 0) - weight_miss
            output_tensor = w.permute(2, 0, 1).reshape(*orig_weight.shape)
        else:
            w = orig_weight.reshape(-1, orig_weight.size(1) // r, r).permute(1, 2, 0) + weight_miss
            output_tensor = w.permute(2, 0, 1).reshape(*orig_weight.shape)

    if cast_to_fp32:
        output_tensor = output_tensor.to(dtype=dtype)
        self.miss_block[adapter].data = weight_miss.to(dtype)

    return output_tensor
```

### Adapter Merging

**Method:** `merge()`

Integrates adapter weights into base layer:

```python
def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
    adapter_names = check_adapters_to_merge(self, adapter_names)
    if not adapter_names:
        return

    for active_adapter in adapter_names:
        if active_adapter in self.miss_block.keys():
            base_layer = self.get_base_layer()
            orig_dtype = base_layer.weight.dtype

            if safe_merge:
                orig_weight = base_layer.weight.data.clone()
                if self.miss_fn == "bat":
                    delta_weight = self.get_delta_weight(active_adapter, orig_weight)
                    orig_weight += delta_weight
                else:
                    delta_weight = self.get_delta_weight_miss(active_adapter, self.base_layer.weight.data)
                    orig_weight = delta_weight

                if not torch.isfinite(orig_weight).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )

                base_layer.weight.data = orig_weight.to(orig_dtype)
            else:
                if self.miss_fn == "bat":
                    delta_weight = self.get_delta_weight(active_adapter, self.base_layer.weight.data)
                    base_layer.weight.data += delta_weight.to(orig_dtype)
                else:
                    delta_weight = self.get_delta_weight_miss(active_adapter, self.base_layer.weight.data)
                    base_layer.weight.data = delta_weight.to(orig_dtype)

            self.merged_adapters.append(active_adapter)
```

### Adapter Unmerging

**Method:** `unmerge()`

Removes merged adapter weights:

```python
def unmerge(self) -> None:
    if not self.merged:
        warnings.warn("Already unmerged. Nothing to do.")
        return

    while len(self.merged_adapters) > 0:
        active_adapter = self.merged_adapters.pop()
        base_layer = self.get_base_layer()
        orig_dtype = base_layer.weight.dtype

        if active_adapter in self.miss_block.keys():
            orig_weight = self.get_base_layer().weight.data.clone()
            if self.miss_fn == "bat":
                delta_weight = self.get_delta_weight(active_adapter, orig_weight, re=True)
            else:
                delta_weight = self.get_delta_weight_miss(active_adapter, orig_weight, re=True)

            base_layer.weight.data = delta_weight.to(orig_dtype)
```

## Technical Details

### Parameter Initialization

Three initialization strategies for different modes:

```python
def reset_miss_parameters(self, adapter_name: str, r):
    """Standard mode: zeros initialization"""
    self.miss_block[adapter_name] = nn.Parameter(torch.zeros(r, self.out_features), requires_grad=True)

def reset_bat_parameters(self, adapter_name: str, r):
    """BAT mode: block-diagonal structure"""
    self.miss_block[adapter_name] = nn.Parameter(torch.zeros(self.out_features // r, r, r), requires_grad=True)

def reset_mini_parameters(self, adapter_name: str, r, mini_r):
    """Mini mode: smallest parameter count"""
    self.miss_block[adapter_name] = nn.Parameter(torch.zeros(r, mini_r), requires_grad=True)
```

### Memory Efficiency

**Cast to FP32 on CPU:**
For CPU inference with FP16/BF16, temporarily cast to FP32 for stable matrix operations:

```python
cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)
if cast_to_fp32:
    weight_miss = weight_miss.float()
# ... perform operations ...
if cast_to_fp32:
    output_tensor = output_tensor.to(dtype=dtype)
```

### Padding Handling

Handles non-divisible input dimensions gracefully:

```python
if x.size(-1) % r != 0:
    padding_size = (r - x.size(-1) % r) % r
    x = F.pad(x, (0, padding_size))
```

## Operational Modes

### 1. Standard MiSS Mode (init_weights=True)

**Parameters:** `r × out_features`

Most efficient and general method. Applies Householder reflections along the input dimension:

```
result = base_output + sum(dropout(x).reshape(..., d/r, r), dim=-2) @ miss_block
```

### 2. BAT Mode (init_weights="bat")

**Parameters:** `(out_features/r) × r × r`

Enables nonlinear block-wise updates. Requires dimensions divisible by r:

```
W' = (W_blocks @ B + B) where B is r×r block matrix
```

### 3. Mini Mode (init_weights="mini")

**Parameters:** `r × mini_r`

Smallest parameter count using repeated patterns:

```
miss_block_expanded = miss_block.repeat(1, out_features // mini_r)
```

## Design Patterns

### Adapter Pattern

Wraps base layers without modifying original architecture:

```python
class MissLinear(nn.Module, MissLayer):
    def __init__(self, base_layer, adapter_name: str, ...):
        super().__init__()
        MissLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
```

### Strategy Pattern

Different computation strategies based on mode:

```python
if self.miss_fn == "bat":
    delta_weight = self.get_delta_weight(active_adapter, orig_weight)
elif self.miss_fn == "mini":
    delta_weight = self.get_delta_weight_miss(active_adapter, orig_weight)
else:
    delta_weight = self.get_delta_weight_miss(active_adapter, orig_weight)
```

## Performance Considerations

### Parameter Efficiency

Comparison with rank r=64, out_features=1024:

- **Standard**: 64 × 1024 = 65,536 parameters
- **BAT**: (1024/64) × 64 × 64 = 65,536 parameters (same, but different structure)
- **Mini** (mini_r=16): 64 × 16 = 1,024 parameters (64x reduction!)

### Computational Efficiency

- **Standard/Mini**: O(d × r) per layer activation
- **BAT**: O(d² / r) for weight transformation (done once or merged)
- Dropout adds minimal overhead
- Padding only when necessary

### Memory Efficiency

- Stores only Householder parameters, not full weight deltas
- Optional weight merging for inference
- CPU FP32 casting only when needed
- No gradient storage when adapters disabled

## Integration Points

### With Base Tuner

Inherits from `BaseTunerLayer` for standard adapter operations:

```python
class MissLayer(BaseTunerLayer):
    adapter_layer_names = ("miss_block",)
    other_param_names = ("miss_r", "miss_dropout", "miss_mini_r")
```

### With PyTorch

Standard `nn.Module` interface:

```python
class MissLinear(nn.Module, MissLayer):
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # Standard forward pass signature
```

## Usage Example

```python
from peft.tuners.miss.layer import MissLinear
import torch.nn as nn

# Create base linear layer
base_layer = nn.Linear(768, 768)

# Wrap with MiSS adapter
miss_layer = MissLinear(
    base_layer,
    adapter_name="default",
    r=64,  # rank for input dimension
    mini_r=1,  # rank for output dimension (1 = standard mode)
    miss_dropout=0.1,
    init_weights=True  # standard MiSS mode
)

# Forward pass
x = torch.randn(32, 128, 768)
output = miss_layer(x)  # Applies Householder reflections

# Merge for efficient inference
miss_layer.merge()
output_merged = miss_layer(x)  # Same result, faster
```

## References

- **Paper**: "MiSS: Minimally Structured Sparse Parameter-Efficient Fine-Tuning" (2024)
- **URL**: https://huggingface.co/papers/2409.15371
- **Key Innovation**: Householder reflections for structured sparsity
- **Efficiency**: Up to 64x parameter reduction vs LoRA with comparable performance
