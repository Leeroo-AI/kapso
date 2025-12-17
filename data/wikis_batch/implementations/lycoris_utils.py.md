# Implementation: lycoris_utils.py

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/lycoris_utils.py`
- **Size**: 263 lines
- **Module**: `peft.tuners.lycoris_utils`
- **Description**: Base classes for LyCORIS-family adapters (LoHa, LoKr, etc.)

## Overview

This module provides abstract base classes for LyCORIS (Lora beyond Conventional methods, Other Rank adaptation Implementations for Stable diffusion) family adapters. LyCORIS methods use alternative matrix factorizations beyond simple low-rank decomposition, such as Hadamard products (LoHa) and Kronecker products (LoKr).

## Core Classes

### LycorisConfig

**Purpose**: Base configuration for LyCORIS adapters

```python
@dataclass
class LycorisConfig(PeftConfig):
    rank_pattern: Optional[dict] = field(default_factory=dict)
    alpha_pattern: Optional[dict] = field(default_factory=dict)
```

**Key Features**:
- `rank_pattern`: Layer-specific rank overrides (regex support)
- `alpha_pattern`: Layer-specific alpha overrides (regex support)
- Extends PeftConfig with pattern matching capabilities

**Example Patterns**:
```python
config = LycorisConfig(
    r=8,  # Default rank
    rank_pattern={
        "^model.decoder.layers.0": 16,  # First decoder layer uses r=16
        ".*attn.*": 12  # All attention layers use r=12
    },
    alpha_pattern={
        "^model.encoder": 16,  # Encoder uses alpha=16
    }
)
```

### LycorisLayer

**Purpose**: Base layer class for LyCORIS adapters

#### Core Attributes

```python
class LycorisLayer(BaseTunerLayer):
    adapter_layer_names: list[str]  # Defined in child classes
    other_param_names = ("r", "alpha", "scaling", "rank_dropout", "module_dropout")

    # Adapter-specific dictionaries
    self.r = {}                    # Rank per adapter
    self.alpha = {}                # Alpha per adapter
    self.scaling = {}              # Computed scaling per adapter
    self.rank_dropout = {}         # Rank dropout per adapter
    self.rank_dropout_scale = {}   # Rank dropout scaling
    self.module_dropout = {}       # Module dropout per adapter
```

#### Abstract Methods

Must be implemented by child classes:

**create_adapter_parameters(adapter_name, r, **kwargs)**:
```python
@abstractmethod
def create_adapter_parameters(self, adapter_name: str, r: int, **kwargs):
    """Initialize adapter-specific weight matrices."""
```

**_get_delta_activations(adapter_name, x, *args, **kwargs)**:
```python
@abstractmethod
def _get_delta_activations(self, adapter_name: str, x: torch.Tensor, ...) -> torch.Tensor:
    """Compute adapter output for given input."""
```

**get_delta_weight(adapter_name)**:
```python
@abstractmethod
def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
    """Compute weight delta for merging."""
```

**reset_adapter_parameters(adapter_name)**:
```python
@abstractmethod
def reset_adapter_parameters(self, adapter_name: str):
    """Re-initialize adapter weights."""
```

**update_layer(adapter_name, r, alpha, **kwargs)**:
```python
@abstractmethod
def update_layer(self, adapter_name: str, r: int, alpha: float, **kwargs):
    """Update adapter hyperparameters."""
```

#### Merge/Unmerge Operations

**merge(safe_merge, adapter_names)**:
```python
def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None):
    """
    Merge adapter weights into base layer.

    Args:
        safe_merge: Check for NaNs before merging
        adapter_names: Specific adapters to merge (None = all active)

    Process:
        1. Get delta weight from adapter
        2. Optionally check for NaNs (safe_merge=True)
        3. Add delta to base layer weights
        4. Track merged adapters
    """
```

**unmerge()**:
```python
def unmerge(self):
    """Subtract merged adapter weights from base layer."""
    while len(self.merged_adapters) > 0:
        active_adapter = self.merged_adapters.pop()
        self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)
```

#### Scaling Operations

**set_scale(adapter, scale)**:
```python
def set_scale(self, adapter, scale):
    """Set adapter scaling: scaling = scale * alpha / r"""
    self.scaling[adapter] = scale * self.alpha[adapter] / self.r[adapter]
```

**scale_layer(scale)**:
```python
def scale_layer(self, scale: float):
    """Multiply all active adapter scalings by factor."""
    for active_adapter in self.active_adapters:
        if active_adapter in self._available_adapters:
            self.scaling[active_adapter] *= scale
```

**unscale_layer(scale)**:
```python
def unscale_layer(self, scale=None):
    """Reset scaling to default (alpha/r) or divide by factor."""
    for active_adapter in self.active_adapters:
        if scale is None:
            self.scaling[active_adapter] = self.alpha[active_adapter] / self.r[active_adapter]
        else:
            self.scaling[active_adapter] /= scale
```

### LycorisTuner

**Purpose**: Base tuner class for LyCORIS models

#### Abstract Method

**_create_and_replace(config, adapter_name, target, target_name, parent, current_key)**:
```python
@abstractmethod
def _create_and_replace(
    self,
    config: LycorisConfig,
    adapter_name: str,
    target: Union[LycorisLayer, nn.Module],
    target_name,
    parent,
    current_key,
):
    """Replace target module with LyCORIS layer."""
```

#### Module Creation

**_create_new_module(config, adapter_name, target, **kwargs)**:
```python
@classmethod
def _create_new_module(cls, config: LycorisConfig, adapter_name: str, target: nn.Module, **kwargs):
    """
    Creates new LycorisLayer from target module.

    Process:
        1. Find matching layer class from layers_mapping
        2. Handle nested tuner layers
        3. Create appropriate layer type (Linear/Conv)
        4. Return configured layer

    Raises:
        ValueError: If target type not supported
    """
```

## Design Patterns

### Empty Weight Initialization

**_init_empty_weights(cls, *args, **kwargs)**:
```python
def _init_empty_weights(self, cls, *args, **kwargs) -> None:
    """
    Initialize layer without materializing weights.

    Inspired by torch.nn.utils.skip_init but compatible with class logic.

    Process:
        1. Set device="meta" in kwargs
        2. Call cls.__init__ (allocates no memory)
        3. Move to target device with to_empty()
    """
```

**Benefits**:
- Faster model loading
- Reduced memory peaks
- Compatible with large model loading strategies

### Pattern Matching

**Rank/Alpha Patterns**:
```python
# Check if current_key matches pattern (regex or exact)
for pattern, custom_rank in config.rank_pattern.items():
    if re.fullmatch(pattern, current_key) or current_key.endswith(pattern):
        r = custom_rank
        break
```

**Use Case**: Different layers need different ranks
- Attention: Higher rank (more expressive)
- MLP: Lower rank (fewer parameters)
- Specific layers: Custom ranks

## LyCORIS Method Examples

### LoHa (Low-rank Hadamard)

**Factorization**:
```
ΔW = (W1_down ⊙ W1_up) @ (W2_down ⊙ W2_up)^T
```
Where ⊙ is Hadamard (element-wise) product.

**Parameters**: 4 matrices vs 2 for LoRA

### LoKr (Low-rank Kronecker)

**Factorization**:
```
ΔW = (W1_left ⊗ W1_right) @ (W2_left ⊗ W2_right)^T
```
Where ⊗ is Kronecker product.

**Benefits**: Efficient for large matrices

## Usage Example

### Implementing Custom LyCORIS Method

```python
class MyLycorisLayer(LycorisLayer):
    adapter_layer_names = ("my_down", "my_up")

    def create_adapter_parameters(self, adapter_name, r, **kwargs):
        # Create custom factorization matrices
        self.my_down[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.my_up[adapter_name] = nn.Linear(r, self.out_features, bias=False)

    def _get_delta_activations(self, adapter_name, x, *args, **kwargs):
        # Compute adapter output
        down = self.my_down[adapter_name]
        up = self.my_up[adapter_name]
        return self.scaling[adapter_name] * up(down(x))

    def get_delta_weight(self, adapter_name):
        # Compute weight delta for merging
        down_weight = self.my_down[adapter_name].weight
        up_weight = self.my_up[adapter_name].weight
        return self.scaling[adapter_name] * (up_weight @ down_weight)
```

## Cross-References

- **Child Classes**: `peft.tuners.loha`, `peft.tuners.lokr`
- **Related**: `peft.tuners.tuners_utils.BaseTuner`, `peft.tuners.tuners_utils.BaseTunerLayer`
- **Used By**: LoHa, LoKr adapter implementations
