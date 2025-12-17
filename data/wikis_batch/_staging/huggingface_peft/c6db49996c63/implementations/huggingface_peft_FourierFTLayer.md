# FourierFTLayer (Sparse Spectral Learning)

## Implementation Overview

**File:** `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/fourierft/layer.py`
**Lines of Code:** 193
**Language:** Python

FourierFTLayer implements parameter-efficient fine-tuning in the frequency domain by learning sparse spectral coefficients. Only a small subset of Fourier frequencies are trained, providing extreme parameter efficiency.

## Core Concept

### Fourier Transform Approach

**Key Idea:** Represent weight updates in frequency domain

1. **Forward:**
   ```
   sparse_spectrum → Dense Spectrum (padding zeros) → IFFT2 → ΔW → Apply
   ```

2. **Trainable:** Only `n_frequency` spectral coefficients
3. **Full weight delta:** Reconstructed via inverse FFT

**Formula:**
```
ΔW = scaling * IFFT2(sparse_spectrum)
W' = W + ΔW
```

## Core Components

### Base Layer Class

```python
class FourierFTLayer(BaseTunerLayer):
    adapter_layer_names = ("fourierft_spectrum",)
    other_param_names = ("fourierft_n_frequency", "fourierft_scaling", "fourierft_random_loc_seed")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.fourierft_n_frequency = {}
        self.fourierft_scaling = {}
        self.fourierft_spectrum = nn.ParameterDict({})
        self.indices = {}  # Random frequency locations
        self.fourierft_random_loc_seed = {}
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            self.in_features, self.out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")
```

### Linear Layer Implementation

```python
class FourierFTLinear(nn.Module, FourierFTLayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        n_frequency: int = 1000,
        scaling: float = 150.0,
        fan_in_fan_out: bool = False,
        init_weights: Union[bool, str] = False,
        random_loc_seed: int = 777,
        **kwargs,
    ) -> None:
        super().__init__()
        FourierFTLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, n_frequency, scaling, init_weights, random_loc_seed)
```

## Key Methods

### Layer Configuration

```python
def update_layer(
    self, adapter_name, n_frequency, scaling, init_weights, random_loc_seed, inference_mode: bool = False, **kwargs
):
    if n_frequency <= 0:
        raise ValueError(f"`n_frequency` should be positive but got {n_frequency}")
    if n_frequency > self.in_features * self.out_features:
        raise ValueError(f"`n_frequency` should be <= {self.in_features * self.out_features}")

    self.fourierft_n_frequency[adapter_name] = n_frequency
    self.fourierft_random_loc_seed[adapter_name] = random_loc_seed

    # Random frequency locations
    self.indices[adapter_name] = torch.randperm(
        self.out_features * self.in_features,
        generator=torch.Generator().manual_seed(random_loc_seed),
    )[:n_frequency]

    # Convert flat indices to 2D coordinates
    self.indices[adapter_name] = torch.stack(
        [self.indices[adapter_name] // self.in_features, self.indices[adapter_name] % self.in_features], dim=0
    )

    self.fourierft_scaling[adapter_name] = scaling

    # Trainable spectral coefficients
    self.fourierft_spectrum[adapter_name] = nn.Parameter(torch.randn(n_frequency), requires_grad=True)

    if init_weights:
        self.reset_fourier_parameters(adapter_name)

    self._move_adapter_to_device_of_base_layer(adapter_name)
    self.set_adapter(self.active_adapters, inference_mode=inference_mode)
```

**Key Steps:**
1. Validate n_frequency
2. Generate random frequency indices
3. Create trainable spectrum parameters
4. Initialize (zeros if init_weights=True)

### Parameter Initialization

```python
@torch.no_grad()
def reset_fourier_parameters(self, adapter_name):
    if adapter_name in self.fourierft_spectrum.keys():
        nn.init.zeros_(self.fourierft_spectrum[adapter_name])
```

**Initialization:**
- `init_weights=True`: Zeros (no adaptation initially)
- `init_weights=False`: Random normal (default)

### Weight Delta Computation

```python
def get_delta_weight(self, adapter) -> torch.Tensor:
    """Compute weight update via inverse FFT"""
    spectrum = self.fourierft_spectrum[adapter]
    indices = self.indices[adapter].to(spectrum.device)

    # Create dense spectrum (zeros everywhere except selected frequencies)
    dense_spectrum = torch.zeros(self.out_features, self.in_features, device=spectrum.device)
    dense_spectrum[indices[0, :], indices[1, :]] = spectrum.float()

    # Inverse FFT2 to get spatial weight delta
    delta_weight = torch.fft.ifft2(dense_spectrum).real * self.fourierft_scaling[adapter]

    return delta_weight.to(spectrum.dtype)
```

**Process:**
1. Create dense spectrum (zeros + sparse coefficients)
2. Apply 2D inverse FFT
3. Take real part (weights are real-valued)
4. Scale by scaling factor
5. Cast back to original dtype

**Important:** IFFT2 requires FP32, so temporarily cast

### Forward Pass

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
        result = self.base_layer(x, *args, **kwargs)
        for active_adapter in self.active_adapters:
            if active_adapter not in self.fourierft_spectrum.keys():
                continue

            delta_w = self.get_delta_weight(active_adapter)
            x = x.to(delta_w.dtype)
            result = result + F.linear(x, delta_w)

    result = result.to(previous_dtype)
    return result
```

**Flow:**
1. Apply base layer
2. For each active adapter:
   - Compute ΔW via IFFT
   - Apply: result += x @ ΔW.T
3. Return result

### Adapter Merging

```python
def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
    adapter_names = check_adapters_to_merge(self, adapter_names)
    if not adapter_names:
        return

    for active_adapter in adapter_names:
        if active_adapter in self.fourierft_spectrum.keys():
            base_layer = self.get_base_layer()

            if safe_merge:
                orig_weights = base_layer.weight.data.clone()
                orig_weights += transpose(self.get_delta_weight(active_adapter), self.fan_in_fan_out)

                if not torch.isfinite(orig_weights).all():
                    raise ValueError(f"NaNs detected in merged weights for {active_adapter}")

                base_layer.weight.data = orig_weights
            else:
                base_layer.weight.data += transpose(self.get_delta_weight(active_adapter), self.fan_in_fan_out)

            self.merged_adapters.append(active_adapter)
```

### Adapter Unmerging

```python
def unmerge(self) -> None:
    if not self.merged:
        warnings.warn("Already unmerged. Nothing to do.")
        return

    while len(self.merged_adapters) > 0:
        active_adapter = self.merged_adapters.pop()
        if active_adapter in self.fourierft_spectrum.keys():
            self.get_base_layer().weight.data -= transpose(
                self.get_delta_weight(active_adapter), self.fan_in_fan_out
            )
```

## Technical Details

### Frequency Selection

**Random Sampling:**
```python
# Generate n_frequency random locations in [0, out_features * in_features)
indices = torch.randperm(out_features * in_features, generator=torch.Generator().manual_seed(seed))[:n_frequency]

# Convert to 2D coordinates
indices_2d = torch.stack([indices // in_features, indices % in_features], dim=0)
```

**Deterministic:** Same seed → same frequencies

### Sparse to Dense Spectrum

```python
# Create dense spectrum (all zeros)
dense_spectrum = torch.zeros(out_features, in_features)

# Fill in sparse coefficients at selected locations
dense_spectrum[indices[0, :], indices[1, :]] = spectrum
```

**Memory:** Only sparse coefficients stored

### Inverse FFT

```python
# IFFT2 requires FP32 for stability
delta_weight = torch.fft.ifft2(dense_spectrum.float()).real * scaling
```

**Properties:**
- IFFT produces complex values
- Take `.real` for real-valued weights
- Scaling applied after IFFT

### Parameter Count

For d×d weight matrix:

**FourierFT:**
- Parameters: `n_frequency`

**LoRA (rank r):**
- Parameters: `2 * d * r`

**Comparison:**
```python
# Example: d=768, r=8, n_frequency=1000
# LoRA: 2 * 768 * 8 = 12,288
# FourierFT: 1,000
# Reduction: 12x fewer parameters!
```

### Computational Cost

**Forward Pass:**
1. Base layer: O(d²)
2. Create dense spectrum: O(n_frequency)
3. IFFT2: O(d² log d)
4. Linear: O(d²)
5. Total: O(d² log d)

**Compared to LoRA:**
- LoRA: O(d²) + O(d·r)
- FourierFT: O(d²) + O(d² log d)
- Slower due to FFT, but mergeable

## Design Patterns

### Lazy Computation

Weight delta computed on-demand:

```python
def forward(self, x):
    delta_w = self.get_delta_weight(adapter)  # Computed per forward
    result = result + F.linear(x, delta_w)
```

### Frequency Domain Processing

```python
# Spatial → Frequency → Spatial
sparse_spectrum → dense_spectrum → IFFT → delta_weight
```

## Usage Example

```python
from peft import FourierFTConfig, get_peft_model

# Configure FourierFT
config = FourierFTConfig(
    n_frequency=1000,        # 1000 spectral coefficients
    scaling=150.0,           # Scaling factor
    random_loc_seed=777,     # Reproducible frequency selection
    target_modules=['q_proj', 'v_proj']
)

# Apply to model
model = get_peft_model(base_model, config)

# Train (gradients only for sparse spectrum)
model.train()
for batch in dataloader:
    output = model(batch)
    loss.backward()  # Only n_frequency gradients!
    optimizer.step()

# Merge for faster inference
model.merge_adapter()
```

## Parameter Efficiency Comparison

For RoBERTa-large (1024-dim layers):

| Method | Parameters per layer | Total (24 layers) |
|--------|---------------------|-------------------|
| LoRA (r=8) | 16,384 | 393K |
| FourierFT (n=1000) | 1,000 | 24K |
| **Reduction** | **16x** | **16x** |

For ViT-large (1024-dim layers):

| Method | Parameters per layer | Total (24 layers) |
|--------|---------------------|-------------------|
| LoRA (r=16) | 32,768 | 786K |
| FourierFT (n=3000) | 3,000 | 72K |
| **Reduction** | **11x** | **11x** |

## References

- **Paper**: "FourierFT: Fourier Transform for Efficient Fine-Tuning" (2024)
- **URL**: https://huggingface.co/papers/2405.03003
- **Key Innovation**: Sparse spectral learning
- **Efficiency**: 10-16x fewer parameters than LoRA
