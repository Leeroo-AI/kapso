# RandLoraLayer (Random Projection LoRA)

## Implementation Overview

**File:** `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/randlora/layer.py`
**Lines of Code:** 350
**Language:** Python

RandLoraLayer implements RandLoRA, a parameter-efficient alternative to LoRA that uses shared random projection matrices across all adapted layers, with only diagonal scaling matrices as trainable parameters. This dramatically reduces parameter count while maintaining performance.

## Core Components

### 1. Memory-Efficient Gradient Function

**Class:** `UniqueBaseGrad(torch.autograd.Function)`

Custom autograd function for efficient gradient computation with shared bases:

```python
class UniqueBaseGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, randlora_A, randlora_lambda, randlora_gamma):
        out = randlora_lambda[:, :, None] * randlora_A * randlora_gamma[None,]
        ctx.save_for_backward(randlora_A, randlora_lambda, randlora_gamma)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        randlora_A, randlora_lambda, randlora_gamma = ctx.saved_tensors
        randlora_A, randlora_lambda, randlora_gamma = (
            randlora_A.to(grad_output.dtype),
            randlora_lambda.to(grad_output.dtype),
            randlora_gamma.to(grad_output.dtype),
        )
        grad_randlora_lambda = torch.einsum("kbj,kvj,bj->kb", grad_output, randlora_A, randlora_gamma)
        grad_randlora_gamma = torch.einsum("kbj,kvj,kb->bj", grad_output, randlora_A, randlora_lambda)
        return None, grad_randlora_lambda, grad_randlora_gamma
```

**Key Features:**
- Only computes gradients for trainable parameters (λ, γ)
- No gradients for frozen random bases (A, B)
- Uses einsum for efficient tensor contractions
- Memory-efficient: doesn't store full scaled matrices

### 2. Base Layer Class

**Class:** `RandLoraLayer(BaseTunerLayer)`

```python
class RandLoraLayer(BaseTunerLayer):
    adapter_layer_names = ("randlora_lambda", "randlora_gamma")
    other_param_names = ("randlora_A", "randlora_B")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.scaling = {}
        self.randlora_dropout = nn.ModuleDict({})

        # Trainable parameters
        self.randlora_lambda = nn.ParameterDict({})
        self.randlora_gamma = nn.ParameterDict({})

        # Non-trainable shared random bases
        self.randlora_A: Optional[BufferDict] = None
        self.randlora_B: Optional[BufferDict] = None

        self._disable_adapters = False
        self.merged_adapters = []
```

**Architecture:**
- **randlora_A, randlora_B**: Shared random projection matrices (frozen)
- **randlora_lambda**: Trainable scaling per rank dimension
- **randlora_gamma**: Trainable scaling per base dimension
- **Shared bases**: Single A/B pair used across all layers

### 3. Linear Layer Implementation

**Class:** `Linear(nn.Linear, RandLoraLayer)`

```python
class Linear(nn.Linear, RandLoraLayer):
    def __init__(
        self,
        base_layer,
        randlora_A: BufferDict,
        randlora_B: BufferDict,
        adapter_name: str,
        r: int = 0,
        randlora_alpha: int = 0,
        randlora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        init_weights: bool = True,
        **kwargs,
    ) -> None:
        super(nn.Linear, self).__init__()
        RandLoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, randlora_A, randlora_B, r, randlora_alpha, randlora_dropout, init_weights)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
```

## Key Methods

### Layer Configuration

**Method:** `update_layer()`

Initializes trainable parameters and references shared bases:

```python
def update_layer(
    self,
    adapter_name,
    randlora_A: BufferDict,
    randlora_B: BufferDict,
    r,
    randlora_alpha,
    randlora_dropout,
    init_weights,
    inference_mode: bool = False,
    **kwargs,
):
    if r <= 0:
        raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

    self.r[adapter_name] = r

    # Dropout
    if randlora_dropout > 0.0:
        randlora_dropout_layer = nn.Dropout(p=randlora_dropout)
    else:
        randlora_dropout_layer = nn.Identity()
    self.randlora_dropout.update(nn.ModuleDict({adapter_name: randlora_dropout_layer}))

    # Calculate number of bases needed for full rank
    num_bases = min(self.in_features, self.out_features) / r
    self.num_bases = int(num_bases) if num_bases.is_integer() else int(num_bases) + 1

    # Trainable parameters
    self.randlora_lambda[adapter_name] = nn.Parameter(torch.randn(r, self.num_bases), requires_grad=True)
    self.randlora_gamma[adapter_name] = nn.Parameter(
        torch.ones(self.num_bases, min(self.out_features, self.in_features))
        / max(self.out_features, self.in_features),
        requires_grad=True,
    )

    self.scaling[adapter_name] = randlora_alpha / r

    # Reference shared bases
    self.randlora_A = randlora_A
    self.randlora_B = randlora_B

    if adapter_name not in randlora_A:
        # Reuse existing bases for new adapters
        randlora_A_param = list(self.randlora_A.values())[0]
        randlora_B_param = list(self.randlora_B.values())[0]

        # Validate dimensions
        max_dim, min_dim = max(self.in_features, self.out_features), min(self.in_features, self.out_features)
        if randlora_B_param.shape[0] < max_dim:
            raise ValueError(f"randlora_B has size {randlora_B_param.shape[0]} but {max_dim} or greater is required")
        if randlora_A_param.shape[-1] < min_dim:
            raise ValueError(f"randlora_A has size {randlora_A_param.shape[1]} but {min_dim} or greater is required")
        if randlora_A_param.shape[0] < self.r[adapter_name]:
            raise ValueError(f"randlora_A has rank {randlora_A_param.shape[0]} but {self.r[adapter_name]} or greater is required")

        self.randlora_A[adapter_name] = randlora_A_param
        self.randlora_B[adapter_name] = randlora_B_param

    if init_weights:
        self.reset_randlora_parameters(adapter_name)
```

**Key Points:**
- Calculates num_bases to ensure full rank capability
- Lambda: shape `[r, num_bases]`
- Gamma: shape `[num_bases, min_dim]`, initialized to small values
- Reuses existing random bases for additional adapters
- Validates dimension compatibility

### Parameter Initialization

**Method:** `reset_randlora_parameters()`

```python
def reset_randlora_parameters(self, adapter_name):
    if adapter_name in self.randlora_lambda.keys():
        with torch.no_grad():
            nn.init.zeros_(self.randlora_lambda[adapter_name])
            nn.init.constant_(self.randlora_gamma[adapter_name], 1 / max(self.randlora_gamma[adapter_name].shape))
```

**Initialization:**
- **Lambda**: All zeros (no adaptation initially)
- **Gamma**: Small constant (1 / max_dim)

### Forward Pass

**Method:** `forward()`

Applies RandLoRA transformation:

```python
def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    previous_dtype = x.dtype

    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        result = self.base_layer(x, *args, **kwargs)
        for active_adapter in self.active_adapters:
            if active_adapter not in self.randlora_lambda.keys():
                continue

            dropout = self.randlora_dropout[active_adapter]
            update_B, update_A = self.get_scaled_bases(active_adapter, device=x.device)
            x = x.to(update_A.dtype)
            scaling = self.scaling[active_adapter]

            # Apply: result += dropout(x) @ update_A.T @ update_B.T * scaling
            result = result + F.linear(F.linear(dropout(x), update_B), update_A) * scaling

    result = result.to(previous_dtype)
    return result
```

**Flow:**
1. Apply base layer transformation
2. Get scaled random bases
3. Apply dropout
4. Compute: x → x @ B.T @ A.T
5. Scale by α/r
6. Add to base output

### Scaled Bases Computation

**Method:** `get_scaled_bases()`

Applies trainable scaling to frozen random bases:

```python
def get_scaled_bases(self, adapter, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    randlora_A = self.randlora_A[adapter]
    randlora_B = self.randlora_B[adapter]
    if device is None:
        device = randlora_B.device
    dtype = randlora_B.dtype

    cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

    randlora_lambda = self.randlora_lambda[adapter].to(device)
    randlora_gamma = self.randlora_gamma[adapter].to(device)

    if cast_to_fp32:
        randlora_A = randlora_A.float()
        randlora_B = randlora_B.float()
        randlora_lambda = randlora_lambda.float()
        randlora_gamma = randlora_gamma.float()

    # Slice required submatrices
    min_dim, max_dim = min(self.out_features, self.in_features), max(self.out_features, self.in_features)
    sliced_A = randlora_A[:, : self.num_bases, :min_dim].to(device)
    sliced_B = randlora_B[:max_dim, : self.num_bases, :].to(device)

    # Apply scaling: A_scaled = lambda * A * gamma
    update_B = sliced_B.flatten(start_dim=1)
    update_A = UniqueBaseGrad.apply(sliced_A, randlora_lambda, randlora_gamma).flatten(end_dim=1)

    # Return in correct order for layer dimensions
    if min_dim == self.in_features:
        return update_A, update_B
    return update_B.T, update_A.T
```

**Operations:**
1. Slice random bases to required dimensions
2. Apply trainable scaling: `A_scaled = λ * A * γ`
3. Flatten over rank and base dimensions
4. Return in correct order based on layer shape

### Weight Delta Computation

**Method:** `get_delta_weight()`

Computes equivalent weight update for merging:

```python
def get_delta_weight(self, adapter) -> torch.Tensor:
    update_B, update_A = self.get_scaled_bases(adapter)
    update = (update_B.T @ update_A.T).T
    output_tensor = transpose(update, self.fan_in_fan_out)
    scaling = self.scaling[adapter]
    return output_tensor * scaling
```

**Formula:** `ΔW = scaling * (B.T @ A.T).T`

### Adapter Merging

**Method:** `merge()`

Integrates adapter into base weights:

```python
def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
    adapter_names = check_adapters_to_merge(self, adapter_names)
    if not adapter_names:
        return

    for active_adapter in adapter_names:
        if active_adapter in self.randlora_lambda.keys():
            base_layer = self.get_base_layer()
            orig_dtype = base_layer.weight.dtype

            if safe_merge:
                orig_weights = base_layer.weight.data.clone()
                orig_weights += self.get_delta_weight(active_adapter)

                if not torch.isfinite(orig_weights).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )

                base_layer.weight.data = orig_weights.to(orig_dtype)
            else:
                delta_weight = self.get_delta_weight(active_adapter)
                base_layer.weight.data += delta_weight.to(orig_dtype)

            self.merged_adapters.append(active_adapter)
```

## Technical Details

### Shared Random Bases

**Key Innovation:**
- Single pair of random matrices (A, B) shared across ALL layers
- Only scaling parameters (λ, γ) are layer-specific
- Massive parameter reduction compared to LoRA

**Base Structure:**
- **randlora_A**: Shape `[r, num_bases, min_dim]`
- **randlora_B**: Shape `[max_dim, num_bases, r]`
- **num_bases**: Calculated to ensure full rank

**Dimension Handling:**
```python
# For different layer sizes, slice from shared bases
sliced_A = randlora_A[:, : self.num_bases, :min_dim]
sliced_B = randlora_B[:max_dim, : self.num_bases, :]
```

### Trainable Parameters

Per adapter, per layer:
- **Lambda (λ)**: `[r, num_bases]` - scales rank dimension
- **Gamma (γ)**: `[num_bases, min_dim]` - scales base dimension

**Parameter Count:**
```python
params_per_layer = r * num_bases + num_bases * min_dim
# For 768-dim layer, r=32, num_bases=24:
# = 32*24 + 24*768 = 768 + 18,432 = 19,200 parameters
# vs LoRA (r=32): 2*768*32 = 49,152 parameters (2.5x more!)
```

### Memory-Efficient Gradients

Using custom autograd for efficient backprop:

```python
# Forward: out = λ[:,:,None] * A * γ[None,:]
# Backward:
# ∂L/∂λ = einsum("kbj,kvj,bj->kb", ∂L/∂out, A, γ)
# ∂L/∂γ = einsum("kbj,kvj,kb->bj", ∂L/∂out, A, λ)
```

**Benefits:**
- No gradients computed for frozen A, B
- Einsum operations are highly optimized
- Memory scales with trainable params only

### Dimension Adaptation

Handles varying layer dimensions:

```python
min_dim, max_dim = min(self.out_features, self.in_features), max(self.out_features, self.in_features)

# Scale applied to smallest dimension
if min_dim == self.in_features:
    return update_A, update_B  # A first (input scaling)
else:
    return update_B.T, update_A.T  # B first (output scaling)
```

## Design Patterns

### Flyweight Pattern

Shared random bases across all layers:

```python
# Single shared bases for all layers
self.randlora_A: Optional[BufferDict] = None  # Shared
self.randlora_B: Optional[BufferDict] = None  # Shared

# Layer-specific trainable parameters
self.randlora_lambda = nn.ParameterDict({})  # Per-layer
self.randlora_gamma = nn.ParameterDict({})   # Per-layer
```

### Strategy Pattern

Different initialization strategies:

```python
if init_weights:
    self.reset_randlora_parameters(adapter_name)  # Zeros
else:
    # Keep random initialization
```

### Custom Autograd Pattern

Specialized gradient computation:

```python
class UniqueBaseGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, randlora_A, randlora_lambda, randlora_gamma):
        # Custom forward

    @staticmethod
    def backward(ctx, grad_output):
        # Custom backward with einsum
```

## Performance Characteristics

### Parameter Efficiency

For a model with N layers of dimension d × d, rank r:

**LoRA:**
- Parameters: `N * 2 * d * r`

**RandLoRA:**
- Shared bases: `r * num_bases * d + d * num_bases * r` (constant)
- Per-layer: `r * num_bases + num_bases * d`
- Total: `~N * (r * num_bases + num_bases * d)`

**Typical Savings:**
- 2-3x fewer parameters than LoRA
- Better with more layers
- Shared bases amortized across layers

### Computational Efficiency

**Forward Pass:**
1. Base layer: O(d²)
2. Dropout: O(d)
3. First projection: O(d * num_bases * r)
4. Second projection: O(num_bases * r * d)
5. Scaling: O(d)

**Total:** O(d²) + O(d * num_bases * r) ≈ O(d²) for small r

### Memory Efficiency

**Training:**
- Shared bases: stored once
- Gradients: only for λ, γ
- No gradients for A, B

**Inference:**
- Can merge adapters
- Or keep separate for multi-task

## Usage Example

```python
from peft.tuners.randlora import RandLoraModel, RandLoraConfig

# Configuration
config = RandLoraConfig(
    r=32,
    randlora_alpha=640,  # Typically 20*r
    target_modules=['q_proj', 'v_proj']
)

# Create model (handles shared base creation)
model = get_peft_model(base_model, config)

# Train
for batch in dataloader:
    output = model(batch)  # Uses shared bases + trainable scaling
    loss.backward()  # Only computes gradients for λ, γ
    optimizer.step()

# Merge for inference
model.merge_adapter()
```

## References

- **Paper**: "RandLoRA: Randomized Low-Rank Adaptation" (2025)
- **URL**: https://huggingface.co/papers/2502.00987
- **Key Innovation**: Shared random projection bases
- **Efficiency**: 2-3x fewer parameters than LoRA
