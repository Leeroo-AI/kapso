# FourierFTModel (FourierFT Model Manager)

## Implementation Overview

**File:** `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/fourierft/model.py`
**Lines of Code:** 128
**Language:** Python

FourierFTModel orchestrates FourierFT adapter creation and management, handling layer replacement and per-layer frequency configuration for sparse spectral learning.

## Core Implementation

```python
class FourierFTModel(BaseTuner):
    """Creates FourierFT model from pretrained transformers model

    Paper: https://huggingface.co/papers/2405.03003

    Args:
        model: Model to adapt
        config: FourierFTConfig
        adapter_name: Adapter name (default: "default")
        low_cpu_mem_usage: Use meta device for initialization

    Returns:
        FourierFT adapted model
    """

    prefix: str = "fourierft_"
    tuner_layer_cls = FourierFTLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_FOURIERFT_TARGET_MODULES_MAPPING
```

## Key Methods

### Layer Creation and Replacement

**Method:** `_create_and_replace()`

Handles per-layer frequency patterns and creates FourierFT layers:

```python
def _create_and_replace(
    self,
    fourierft_config,
    adapter_name,
    target,
    target_name,
    parent,
    current_key,
    **optional_kwargs,
):
    if current_key is None:
        raise ValueError("Current Key shouldn't be `None`")

    # Check for per-layer frequency override
    pattern_keys = list(chain(fourierft_config.n_frequency_pattern.keys()))
    target_name_key = next(
        filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys),
        current_key
    )

    n_frequency = fourierft_config.n_frequency_pattern.get(
        target_name_key,
        fourierft_config.n_frequency
    )
    scaling = fourierft_config.scaling
    random_loc_seed = fourierft_config.random_loc_seed

    bias = hasattr(target, "bias") and target.bias is not None
    kwargs = {
        "n_frequency": n_frequency,
        "scaling": scaling,
        "fan_in_fan_out": fourierft_config.fan_in_fan_out,
        "init_weights": fourierft_config.init_weights,
        "random_loc_seed": fourierft_config.random_loc_seed,
    }
    kwargs["bias"] = bias

    if isinstance(target, FourierFTLayer):
        # Update existing layer
        target.update_layer(
            adapter_name,
            n_frequency,
            scaling,
            fourierft_config.init_weights,
            random_loc_seed,
        )
    else:
        # Create new layer
        new_module = self._create_new_module(fourierft_config, adapter_name, target, **kwargs)
        if adapter_name != self.active_adapter:
            new_module.requires_grad_(False)
        self._replace_module(parent, target_name, new_module, target)
```

**Key Features:**
1. Regex matching for per-layer frequency patterns
2. Falls back to default n_frequency if no match
3. Handles existing FourierFT layers (multi-adapter)
4. Creates new layers for first adapter

### Module Factory

**Method:** `_create_new_module()`

Creates FourierFTLinear instances with proper configuration:

```python
@staticmethod
def _create_new_module(fourierft_config, adapter_name, target, **kwargs):
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = fourierft_config.fan_in_fan_out = False
    elif isinstance(target_base_layer, Conv1D):
        kwargs["is_target_conv_1d_layer"] = True
        if not kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to False but target module is `Conv1D`. "
                "Setting fan_in_fan_out to True."
            )
            kwargs["fan_in_fan_out"] = fourierft_config.fan_in_fan_out = True
    else:
        raise ValueError(
            f"Target module {target} not supported. "
            "Currently, only `torch.nn.Linear` supported."
        )

    new_module = FourierFTLinear(target, adapter_name, **kwargs)

    return new_module
```

**Supported Layers:**
- `torch.nn.Linear`: Standard dense layers
- `Conv1D`: GPT-2 style layers (with fan_in_fan_out=True)

## Per-Layer Frequency Configuration

### Regex Matching

The model supports flexible per-layer frequency patterns via regex:

```python
# Config example
config = FourierFTConfig(
    n_frequency=1000,  # Default
    n_frequency_pattern={
        'encoder.layer.0.attention.self.query': 500,     # Exact match
        'encoder.layer.\d+.attention.self.value': 1500,  # Regex match
    }
)

# Matching logic
pattern_keys = list(config.n_frequency_pattern.keys())
target_name_key = next(
    filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys),
    current_key
)
n_frequency = config.n_frequency_pattern.get(target_name_key, config.n_frequency)
```

**Matching Priority:**
1. Check if current_key matches any pattern key
2. Use matched pattern's n_frequency
3. Fall back to default config.n_frequency

### Example Usage

```python
from peft import FourierFTConfig, get_peft_model

# Configure with per-layer frequencies
config = FourierFTConfig(
    n_frequency=1000,  # Default for most layers
    scaling=150.0,
    target_modules='all-linear',
    n_frequency_pattern={
        # Lower frequencies for early layers
        'encoder.layer.0': 500,
        'encoder.layer.1': 500,
        # Higher frequencies for late layers
        'encoder.layer.10': 2000,
        'encoder.layer.11': 2000,
        # Specific attention heads
        'encoder.layer.*.attention.self.query': 1500,
    }
)

# Apply to model
model = get_peft_model(base_model, config)

# Each layer gets appropriate n_frequency based on pattern matching
```

## Architecture Support

### Pre-defined Target Modules

```python
target_module_mapping = TRANSFORMERS_MODELS_TO_FOURIERFT_TARGET_MODULES_MAPPING

# Example mappings:
# BERT: ["query", "key", "value"]
# RoBERTa: ["query", "key", "value"]
# GPT-2: ["c_attn", "c_proj"]
# LLaMA: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

### Auto-detection

If `target_modules` not specified, uses architecture-specific defaults.

## Usage Examples

### Basic Usage

```python
from transformers import AutoModel
from peft import FourierFTConfig, get_peft_model

# Load base model
base_model = AutoModel.from_pretrained("bert-base-uncased")

# Create FourierFT configuration
config = FourierFTConfig(
    n_frequency=1000,
    scaling=150.0,
    target_modules=['query', 'value']
)

# Apply FourierFT
model = get_peft_model(base_model, config)

# Train
model.train()
for batch in dataloader:
    output = model(batch)
    loss.backward()  # Gradients only for n_frequency params!
    optimizer.step()
```

### Per-Layer Configuration

```python
config = FourierFTConfig(
    n_frequency=1000,
    scaling=150.0,
    target_modules='all-linear',
    n_frequency_pattern={
        # Pattern-based configuration
        'encoder.layer.[0-3]': 500,      # First 4 layers: 500 frequencies
        'encoder.layer.[4-7]': 1000,     # Middle layers: 1000 frequencies
        'encoder.layer.[8-11]': 2000,    # Last 4 layers: 2000 frequencies
    }
)

model = get_peft_model(base_model, config)
```

### Multi-Adapter Usage

```python
# Add first adapter
config1 = FourierFTConfig(n_frequency=1000, scaling=150.0)
model = get_peft_model(base_model, config1, adapter_name="task1")

# Add second adapter
config2 = FourierFTConfig(n_frequency=2000, scaling=200.0)
model.add_adapter("task2", config2)

# Switch between adapters
model.set_adapter("task1")
output1 = model(batch)

model.set_adapter("task2")
output2 = model(batch)
```

### Merging for Inference

```python
# Training mode
model.train()
for batch in dataloader:
    output = model(batch)  # Computes IFFT per forward
    loss.backward()
    optimizer.step()

# Inference mode: merge adapters for speed
model.eval()
model.merge_adapter()
output = model(batch)  # No IFFT computation, just matrix multiply!
```

## Design Patterns

### Strategy Pattern

Different layer types handled via dispatchers:

```python
if isinstance(target_base_layer, torch.nn.Linear):
    # Handle Linear
elif isinstance(target_base_layer, Conv1D):
    # Handle Conv1D
else:
    raise ValueError("Unsupported")
```

### Template Method Pattern

```python
class FourierFTModel(BaseTuner):
    # Inherits: inject_adapter, merge_adapter, save_pretrained
    # Overrides: _create_and_replace, _create_new_module
```

### Factory Pattern

```python
@staticmethod
def _create_new_module(config, adapter_name, target, **kwargs):
    # Creates appropriate FourierFT module
    return FourierFTLinear(target, adapter_name, **kwargs)
```

## Performance Characteristics

### Parameter Efficiency

For 24-layer BERT-base (768-dim):

**LoRA (r=8):**
- Params per layer: 12,288
- Total: 294K parameters

**FourierFT (n=1000):**
- Params per layer: 1,000
- Total: 24K parameters
- **Reduction: 12.3x**

### Computational Efficiency

**Training:**
- Forward: O(d² log d) due to IFFT
- Backward: O(n_frequency) gradients only
- Slower than LoRA due to FFT

**Inference (merged):**
- Same as base model
- No FFT overhead
- Mergeable like LoRA

**Inference (unmerged):**
- Extra IFFT per forward
- Extra matrix multiply
- Slower than merged

**Recommendation:** Merge adapters for inference

## References

- **Paper**: https://huggingface.co/papers/2405.03003
- **Type**: `PeftType.FOURIERFT`
- **Prefix**: "fourierft_"
- **Key Feature**: Per-layer frequency configuration
- **Efficiency**: 10-20x fewer parameters than LoRA
