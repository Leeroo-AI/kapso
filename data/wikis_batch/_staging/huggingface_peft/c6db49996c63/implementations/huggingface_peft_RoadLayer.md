# RoadLayer (2D Rotation Adaptation)

## Implementation Overview

**File:** `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/road/layer.py`
**Lines of Code:** 418
**Language:** Python

RoadLayer implements RoAd (Rotation Adaptation), which adapts models by applying learned 2D rotations to pairs of elements in the hidden representations. This method stores only rotation angles (θ) and scales (α), enabling extremely parameter-efficient adaptation.

## Core Concept

### 2D Rotation Mathematics

**Basic Rotation:** Each pair of elements `(x₀, x₁)` is rotated:

```
y₀ = x₀ * α * cos(θ) - x₁ * α * sin(θ)
y₁ = x₀ * α * sin(θ) + x₁ * α * cos(θ)
```

**Matrix Form:**
```
[y₀]   [α*cos(θ)  -α*sin(θ)] [x₀]
[y₁] = [α*sin(θ)   α*cos(θ)] [x₁]
```

**Grouping Strategy:** Elements are grouped and paired:
- Element 0 pairs with element group_size/2
- Element 1 pairs with element group_size/2 + 1
- Etc.

## Core Components

### 1. Base Layer Class

**Class:** `RoadLayer(BaseTunerLayer)`

```python
class RoadLayer(BaseTunerLayer):
    adapter_layer_names: tuple[str, ...] = ("road_theta", "road_alpha")
    other_param_names: tuple[str, ...] = ("variant", "group_size")

    def __init__(self, base_layer: nn.Module, ephemeral_gpu_offload: bool = False, **kwargs) -> None:
        self.base_layer = base_layer
        self.variant = {}  # road_1, road_2, or road_4
        self.group_size = {}
        self.road_theta = nn.ParameterDict({})  # Rotation angles
        self.road_alpha = nn.ParameterDict({})  # Scales

        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            raise ValueError(f"Unsupported layer type '{type(base_layer)}'")

        self.in_features = in_features
        self.out_features = out_features
```

**Key Attributes:**
- `road_theta`: Rotation angles (trainable)
- `road_alpha`: Scale factors (trainable)
- `variant`: Parameter sharing mode
- `group_size`: Grouping for rotation pairs

### 2. Linear Layer Implementation

**Class:** `Linear(nn.Module, RoadLayer)`

```python
class Linear(nn.Module, RoadLayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        variant: RoadVariant = "road_1",
        group_size: int = 64,
        init_weights: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        RoadLayer.__init__(self, base_layer, **kwargs)

        self._active_adapter = adapter_name

        self.update_layer(
            adapter_name,
            variant,
            group_size,
            init_weights=init_weights,
        )
```

## Variants

### Variant Comparison

**road_1:** Minimal parameters (out_features / 2)
- Same θ and α for each pair of elements in a group
- Parameters per layer: `out_features // 2`

**road_2:** Moderate parameters (out_features)
- Unique θ and α for each element
- Parameters per layer: `out_features`

**road_4:** Maximum parameters (out_features * 2)
- Separate θ and α for cos and sin components
- Parameters per layer: `out_features * 2`

## Key Methods

### Layer Configuration

**Method:** `update_layer()`

```python
def update_layer(
    self,
    adapter_name,
    variant,
    group_size,
    init_weights,
    inference_mode: bool = False,
):
    self.variant[adapter_name] = variant
    self.group_size[adapter_name] = group_size

    if self.out_features % group_size != 0:
        raise ValueError(
            f"out_features must be divisible by group_size ({group_size})"
        )

    # Determine parameter count based on variant
    if variant == "road_1":
        size = self.out_features // 2
    elif variant == "road_2":
        size = self.out_features
    elif variant == "road_4":
        size = self.out_features * 2
    else:
        raise ValueError(f"Unsupported variant {variant}")

    self.road_theta[adapter_name] = nn.Parameter(torch.empty(size))
    self.road_alpha[adapter_name] = nn.Parameter(torch.empty(size))

    self.reset_parameters(adapter_name, init_weights)
    self._move_adapter_to_device_of_base_layer(adapter_name)
    self.set_adapter(self.active_adapters, inference_mode=inference_mode)
```

### Parameter Initialization

**Method:** `reset_parameters()`

```python
def reset_parameters(self, adapter_name, init_weights):
    if init_weights is False:
        # Random initialization
        nn.init.normal_(self.road_theta[adapter_name].data, mean=0.0, std=0.5)
        nn.init.normal_(self.road_alpha[adapter_name].data, mean=1.0, std=0.5)
        return

    # Default: Identity transformation
    nn.init.zeros_(self.road_theta[adapter_name].data)  # θ=0 → no rotation
    nn.init.ones_(self.road_alpha[adapter_name].data)   # α=1 → preserve magnitude
```

**Initialization:**
- `init_weights=True`: Identity (θ=0, α=1)
- `init_weights=False`: Random (θ~N(0,0.5), α~N(1,0.5))

### Forward Pass

**Method:** `forward()`

```python
def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    self._check_forward_args(x, *args, **kwargs)
    adapter_names = kwargs.pop("adapter_names", None)

    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    elif adapter_names is not None:
        result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
    else:
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        for active_adapter in self.active_adapters:
            if active_adapter not in self._available_adapters:
                continue

            result = self._cast_input_dtype(result, self.road_theta[active_adapter].dtype)
            result = _apply_road(
                self.variant[active_adapter],
                self.group_size[active_adapter],
                self.road_theta[active_adapter],
                self.road_alpha[active_adapter],
                result,
            )

        result = result.to(torch_result_dtype)

    return result
```

### Rotation Application

**Function:** `_apply_road()`

Applies 2D rotations efficiently:

```python
def _apply_road(
    variant: RoadVariant,
    group_size: int,
    road_theta: torch.Tensor,
    road_alpha: torch.Tensor,
    x: torch.Tensor
):
    first_col, second_col = _prepare_cols(variant, group_size, road_theta, road_alpha)

    # Split in half groups and join back
    # Equation 4 in RoAD paper
    x_grouped = x.reshape(-1, 2, group_size // 2)
    x1 = x_grouped[:, 0, :]
    x2 = x_grouped[:, 1, :]
    rotate_half_x = torch.stack((-x2, x1), dim=1).reshape(x.shape)
    result = x * first_col + rotate_half_x * second_col
    return result
```

**Formula:**
```
result = x * (α * cos(θ)) + rotate_half(x) * (α * sin(θ))
```

Where `rotate_half(x)` swaps and negates paired elements.

### Column Preparation

**Function:** `_prepare_cols()`

Prepares cos/sin columns based on variant:

```python
def _prepare_cols(
    variant: RoadVariant,
    group_size: int,
    road_theta: torch.Tensor,
    road_alpha: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if variant == "road_1":
        # Reuse parameters within each group
        road_theta = road_theta.reshape(-1, group_size // 2).repeat_interleave(2, dim=0).flatten()
        road_alpha = road_alpha.reshape(-1, group_size // 2).repeat_interleave(2, dim=0).flatten()

        theta_cos = road_theta.cos()
        theta_sin = road_theta.sin()

        first_col = road_alpha * theta_cos
        second_col = road_alpha * theta_sin

    elif variant == "road_2":
        # Unique parameters per element
        theta_cos = road_theta.cos()
        theta_sin = road_theta.sin()

        first_col = road_alpha * theta_cos
        second_col = road_alpha * theta_sin

    elif variant == "road_4":
        # Separate parameters for cos and sin
        road_theta = road_theta.reshape(-1, 2, group_size)
        theta_cos = road_theta[:, 0, :].cos().flatten()
        theta_sin = road_theta[:, 1, :].sin().flatten()

        road_alpha = road_alpha.reshape(-1, 2, group_size)
        alpha_1 = road_alpha[:, 0, :].flatten()
        alpha_2 = road_alpha[:, 1, :].flatten()

        first_col = alpha_1 * theta_cos
        second_col = alpha_2 * theta_sin

    return first_col, second_col
```

### Weight Delta Computation

**Function:** `_get_delta_weight()`

Computes rotation matrix for merging:

```python
def _get_delta_weight(
    variant: RoadVariant,
    group_size: int,
    road_theta: torch.Tensor,
    road_alpha: torch.Tensor
):
    first_col, second_col = _prepare_cols(variant, group_size, road_theta, road_alpha)

    # First column on main diagonal
    output_tensor = torch.diag(first_col)

    # Second column on rotated diagonal (like RoPE embeddings)
    size = second_col.shape[0]
    swapped_second_col = second_col.reshape(-1, 2, group_size // 2)[:, [1, 0], :].flatten()
    rotated_diag_second_col = torch.diag(swapped_second_col).reshape(-1, 2, group_size // 2, size)[:, [1, 0], :, :]
    rotated_diag_second_col[:, 0, :, :] *= -1
    rotated_diag_second_col = rotated_diag_second_col.reshape(size, size)
    output_tensor += rotated_diag_second_col

    return output_tensor
```

**Result:** Block-diagonal rotation matrix R

### Adapter Merging

**Method:** `merge()`

Merges rotation into base weights:

```python
def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
    adapter_names = check_adapters_to_merge(self, adapter_names)
    if not adapter_names:
        return

    for active_adapter in adapter_names:
        if active_adapter in self._available_adapters:
            base_layer = self.get_base_layer()
            orig_dtype = base_layer.weight.dtype

            road_R = _get_delta_weight(
                self.variant[active_adapter],
                self.group_size[active_adapter],
                self.road_theta[active_adapter].data,
                self.road_alpha[active_adapter].data,
            )

            if safe_merge:
                orig_weight = base_layer.weight.data.clone()
                orig_weight = torch.matmul(road_R.to(orig_dtype), orig_weight)

                if not torch.isfinite(orig_weight).all():
                    raise ValueError(f"NaNs detected in merged weights")

                base_layer.weight.data = orig_weight.contiguous().to(orig_dtype)

                if base_layer.bias is not None:
                    orig_bias = base_layer.bias.clone()
                    orig_bias = torch.matmul(road_R.to(orig_dtype), orig_bias)

                    if not torch.isfinite(orig_bias).all():
                        raise ValueError(f"NaNs detected in merged bias")

                    base_layer.bias.data = orig_bias.contiguous().to(orig_dtype)
            else:
                # Direct merge
                orig_weight = base_layer.weight.data
                orig_weight = torch.matmul(road_R.to(orig_dtype), orig_weight)
                base_layer.weight.data = orig_weight.contiguous().to(orig_dtype)

                if base_layer.bias is not None:
                    orig_bias = base_layer.bias.data
                    orig_bias = torch.matmul(road_R.to(orig_dtype), orig_bias)
                    base_layer.bias.data = orig_bias.contiguous().to(orig_dtype)

            self.merged_adapters.append(active_adapter)
```

**Merge Formula:**
```
W' = R @ W
b' = R @ b
```

### Adapter Unmerging

**Method:** `unmerge()`

Removes merged rotation:

```python
def unmerge(self) -> None:
    if not self.merged:
        warnings.warn("Already unmerged. Nothing to do.")
        return

    while len(self.merged_adapters) > 0:
        active_adapter = self.merged_adapters.pop()
        if active_adapter in self._available_adapters:
            weight = self.get_base_layer().weight
            orig_dtype = weight.dtype

            road_R = _get_delta_weight(
                self.variant[active_adapter],
                self.group_size[active_adapter],
                self.road_theta[active_adapter].data,
                self.road_alpha[active_adapter].data,
            )

            # Use inverse (not transpose, as matrix may not be orthogonal)
            inv_road_R = torch.linalg.inv(road_R.to(torch.float32)).to(orig_dtype)
            orig_weight = torch.matmul(inv_road_R, weight.data)
            weight.data = orig_weight.contiguous()

            if self.get_base_layer().bias is not None:
                orig_bias = torch.matmul(inv_road_R, self.get_base_layer().bias.data)
                self.get_base_layer().bias.data = orig_bias.contiguous()
```

**Important:** Uses matrix inverse (not transpose) since R may not be orthogonal.

### Mixed Batch Forward

**Method:** `_mixed_batch_forward()`

Applies different adapters to different samples in batch:

```python
def _mixed_batch_forward(
    self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
) -> torch.Tensor:
    result = self.base_layer(x, *args, **kwargs)

    unique_adapters = set(adapter_names)
    sub_batch_indices_list = []
    for adapter in unique_adapters:
        sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

    for i, active_adapter in enumerate(unique_adapters):
        if active_adapter == "__base__":
            continue
        if active_adapter not in self._available_adapters:
            continue

        dtype = self.road_theta[active_adapter].data.dtype

        # Apply rotation to sub-batch
        sub_batch = result[sub_batch_indices_list[i]].to(dtype)
        result[sub_batch_indices_list[i]] = _apply_road(
            self.variant[active_adapter],
            self.group_size[active_adapter],
            self.road_theta[active_adapter],
            self.road_alpha[active_adapter],
            sub_batch,
        )

    return result
```

**Use Case:** Multi-task inference with different adapters per sample

## Parameter Efficiency

### Parameter Counts

For a 768-dimensional layer:

**road_1:**
- Parameters: 768 / 2 = 384
- Stores: 192 θ + 192 α

**road_2:**
- Parameters: 768
- Stores: 384 θ + 384 α

**road_4:**
- Parameters: 768 * 2 = 1,536
- Stores: 768 θ + 768 α

**Comparison to LoRA (r=8):**
- LoRA: 2 * 768 * 8 = 12,288
- road_1: 384 (32x fewer!)
- road_2: 768 (16x fewer!)
- road_4: 1,536 (8x fewer!)

### Computational Efficiency

**Forward Pass:**
- Base layer: O(d²)
- Rotation: O(d) element-wise ops
- Total: O(d²) dominated by base layer

**Merge:**
- Compute R: O(d²) sparse matrix
- Merge: O(d²) matrix multiply
- One-time cost, fast inference after

## Design Patterns

### Template Method Pattern

```python
def forward(self, x, *args, **kwargs):
    result = self.base_layer(x)  # Template
    result = _apply_road(...)     # Variant-specific
    return result
```

### Strategy Pattern

Different variants use different parameter sharing:

```python
if variant == "road_1":
    # Minimal params
elif variant == "road_2":
    # Moderate params
elif variant == "road_4":
    # Maximum params
```

## Usage Example

```python
from peft import RoadConfig, get_peft_model

# Configure RoAd
config = RoadConfig(
    variant="road_1",     # Minimal parameters
    group_size=64,        # Grouping size
    target_modules=['q_proj', 'v_proj']
)

# Apply to model
model = get_peft_model(base_model, config)

# Train
model.train()

# Merge for fast inference
model.merge_adapter()
```

## References

- **Paper**: "RoAd: Rotation Adaptation for Efficient Fine-Tuning" (2024)
- **URL**: https://huggingface.co/papers/2409.00119
- **Key Innovation**: 2D rotation matrices for adaptation
- **Efficiency**: 8-32x fewer parameters than LoRA
