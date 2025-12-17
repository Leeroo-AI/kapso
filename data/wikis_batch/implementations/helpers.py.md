# Implementation: helpers.py

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/helpers.py`
- **Size**: 251 lines
- **Module**: `peft.helpers`
- **Description**: User-facing helper functions for common PEFT operations

## Overview

This module provides high-level utilities that simplify common PEFT workflows, including method signature updates, model type detection, adapter scaling manipulation, and input dtype casting control. These helpers bridge the gap between the low-level API and user convenience.

## Core Functions

### Signature Update Utilities

**update_forward_signature(model: PeftModel) → None**
```python
def update_forward_signature(model: PeftModel) -> None:
    """
    Updates the forward signature to include parent class signature.

    Problem: PeftModel.forward has generic (*args, **kwargs) signature
    Solution: Copy signature from base model's forward for better IDE support

    Process:
    1. Check if current signature is generic (*args, **kwargs)
    2. Copy __doc__, __name__, __annotations__ from base model
    3. Bind as method to PeftModel instance
    """
```

**update_generate_signature(model: PeftModel) → None**
```python
def update_generate_signature(model: PeftModel) → None:
    """
    Updates the generate signature to include parent class signature.

    Use Case: Generation-capable models (GPT, T5, BART)
    Benefit: Proper IDE autocomplete for generate() arguments
    """
```

**update_signature(model: PeftModel, method: str = "all") → None**
```python
def update_signature(model: PeftModel, method: str = "all") -> None:
    """
    Unified signature updater.

    Args:
        model: PeftModel instance
        method: "forward", "generate", or "all"

    Raises:
        ValueError: If method not in ["forward", "generate", "all"]
    """
```

#### Implementation Details

**Signature Detection**:
```python
current_signature = inspect.signature(model.forward)
if (
    len(current_signature.parameters) == 2
    and "args" in current_signature.parameters
    and "kwargs" in current_signature.parameters
):
    # Generic signature detected, update needed
```

**Signature Copying**:
```python
forward = deepcopy(model.forward.__func__)
update_wrapper(
    forward,
    type(model.get_base_model()).forward,
    assigned=("__doc__", "__name__", "__annotations__")
)
model.forward = MethodType(forward, model)
```

### Model Type Detection

**check_if_peft_model(model_name_or_path: str) → bool**
```python
def check_if_peft_model(model_name_or_path: str) -> bool:
    """
    Checks if model is a PEFT model by attempting config load.

    Process:
    1. Try loading PeftConfig from path
    2. If successful → PEFT model
    3. If exception → Not PEFT model

    Use Cases:
    - Conditional loading logic
    - Auto-detection in pipelines
    - Validation before operations

    Robust: Catches all exceptions (future-proof for new HF Hub errors)
    """
```

**Example Usage**:
```python
if check_if_peft_model("my-username/my-adapter"):
    model = PeftModel.from_pretrained(base_model, "my-username/my-adapter")
else:
    model = AutoModel.from_pretrained("my-username/my-adapter")
```

### Adapter Scaling

**rescale_adapter_scale(model, multiplier) [Context Manager]**
```python
@contextmanager
def rescale_adapter_scale(model, multiplier):
    """
    Temporarily rescales LoRA adapter scaling factors.

    Mathematical Effect:
        output_scaled = output_base + (multiplier * scaling) * (B @ A @ x)

    Use Cases:
    1. Wise-FT: Interpolate between base and adapted model
    2. Distribution shift: Reduce adapter influence
    3. Ablation studies: Scale adapter contribution

    Args:
        model: Model containing LoraLayer modules
        multiplier (float/int): Scaling factor (typically 0-1)

    Raises:
        TypeError: If multiplier not float/int
        ValueError: If no LoraLayer found in model
    """
```

#### Implementation

**Scaling Application**:
```python
original_scaling = {}
for module in model.modules():
    if isinstance(module, LoraLayer):
        original_scaling[module] = module.scaling.copy()
        module.scaling = {k: v * multiplier for k, v in module.scaling.items()}
```

**Automatic Restoration**:
```python
try:
    yield
finally:
    # Restore original scaling after context exit
    for module, scaling in original_scaling.items():
        module.scaling = scaling
```

#### Wise-FT Equivalence

For multiplier ∈ [0, 1], this implements [Wise-FT](https://huggingface.co/papers/2109.01903):

```
W_wiseft = α * W_base + (1-α) * W_adapted
         = W_base + (1-α) * (W_adapted - W_base)
         = W_base + (1-α) * LoRA_delta
```

Setting `multiplier = 1-α` achieves this interpolation.

**Example**:
```python
# Reduce adapter influence by 50%
with rescale_adapter_scale(model, 0.5):
    outputs = model(**inputs)
# Original scaling restored here
```

**Warning**: On Apple MPS backend, add short sleep after context exit for full restoration.

### Input Dtype Casting Control

**disable_input_dtype_casting(model: nn.Module, active: bool = True) [Context Manager]**
```python
@contextmanager
def disable_input_dtype_casting(model: nn.Module, active: bool = True):
    """
    Disables automatic input dtype casting in PEFT layers.

    Normal Behavior:
        BaseTunerLayer.forward casts inputs to weight.dtype

    Use Case:
        Layerwise casting in diffusers conflicts with PEFT casting
        This allows external hooks to control dtype

    Args:
        model: Model with BaseTunerLayer modules
        active: Whether context is active (default True)
    """
```

#### Implementation

**Tracking Original State**:
```python
original_values = {}
for name, module in model.named_modules():
    if not isinstance(module, BaseTunerLayer):
        continue
    original_values[name] = module.cast_input_dtype_enabled
    module.cast_input_dtype_enabled = False
```

**Restoration**:
```python
try:
    yield
finally:
    for name, module in model.named_modules():
        if not isinstance(module, BaseTunerLayer):
            continue
        if name in original_values:
            module.cast_input_dtype_enabled = original_values[name]
```

**Example**:
```python
# Diffusers layerwise casting
with disable_input_dtype_casting(model):
    # Forward hook handles dtype conversion
    outputs = model(**inputs)
# PEFT casting re-enabled
```

## Advanced Use Cases

### Wise-FT with Adapter Scaling

```python
import torch
from peft.helpers import rescale_adapter_scale

# Test multiple interpolation points
multipliers = [0.0, 0.25, 0.5, 0.75, 1.0]
results = []

for mult in multipliers:
    with rescale_adapter_scale(model, mult):
        outputs = model(test_inputs)
        metric = evaluate(outputs, targets)
        results.append((mult, metric))

# Find optimal interpolation
best_mult = max(results, key=lambda x: x[1])[0]
```

### Distribution Shift Adaptation

```python
# Training data distribution != test data distribution
# Reduce adapter influence on OOD data

with rescale_adapter_scale(model, multiplier=0.7):
    # Less aggressive adaptation for OOD inputs
    ood_outputs = model(ood_inputs)
```

### Signature Update for IDE Support

```python
from peft import get_peft_model, LoraConfig
from peft.helpers import update_signature

config = LoraConfig(...)
peft_model = get_peft_model(model, config)

# Enable proper IDE autocomplete
update_signature(peft_model, method="all")

# Now IDE shows proper signature
peft_model.forward(input_ids=..., attention_mask=...)  # Autocompletes!
peft_model.generate(input_ids=..., max_length=...)     # Autocompletes!
```

## Design Patterns

### Context Manager Pattern

**Benefits**:
1. **Automatic Cleanup**: Always restores state
2. **Exception Safety**: Cleanup happens even on error
3. **Readability**: Clear scope of temporary changes

**Example**:
```python
# Without context manager (error-prone)
original = model.scaling.copy()
try:
    model.scaling *= 0.5
    outputs = model(**inputs)
finally:
    model.scaling = original

# With context manager (clean)
with rescale_adapter_scale(model, 0.5):
    outputs = model(**inputs)
```

### Signature Preservation

**Motivation**: PeftModel wraps base model, losing type hints

**Solution**: Copy metadata from base model
- `__doc__`: Documentation strings
- `__name__`: Function name
- `__annotations__`: Type hints

**Benefit**: IDEs and type checkers work correctly

## Performance Considerations

### Signature Update
- **Cost**: O(1) one-time operation
- **When**: Call once after model creation
- **Impact**: None at runtime (only affects IDE/docs)

### Adapter Scaling
- **Cost**: O(num_adapters) dictionary operations
- **When**: Context enter/exit
- **Impact**: Negligible (< 1ms for typical models)

### Dtype Casting Control
- **Cost**: O(num_peft_layers) attribute access
- **When**: Context enter/exit
- **Impact**: Negligible

## Error Handling

### rescale_adapter_scale Errors

```python
# Invalid multiplier type
rescale_adapter_scale(model, "0.5")
# TypeError: Argument multiplier should be of type float, got <class 'str'>

# No LoRA layers
rescale_adapter_scale(base_model, 0.5)
# ValueError: scaling is only supported for models with `LoraLayer`s
```

### update_signature Errors

```python
# Invalid method
update_signature(model, method="invalid")
# ValueError: method invalid is not supported, choose one of ['forward', 'generate', 'all']
```

## Integration Examples

### Transformers Pipeline

```python
from transformers import pipeline
from peft import PeftModel
from peft.helpers import update_signature, check_if_peft_model

model_id = "my-model"
if check_if_peft_model(model_id):
    base = AutoModelForCausalLM.from_pretrained("base-model")
    model = PeftModel.from_pretrained(base, model_id)
    update_signature(model)
else:
    model = AutoModelForCausalLM.from_pretrained(model_id)

pipe = pipeline("text-generation", model=model)
```

### Diffusers Compatibility

```python
from diffusers import StableDiffusionPipeline
from peft.helpers import disable_input_dtype_casting

pipeline = StableDiffusionPipeline.from_pretrained(...)

# Diffusers uses layerwise casting hooks
with disable_input_dtype_casting(pipeline.unet):
    images = pipeline(prompt)
```

## Cross-References

- **Related**: `peft.functional` (low-level API), `peft.peft_model` (PeftModel class)
- **Dependencies**: `torch.nn`, `inspect`, `contextlib`, `copy`, `types`
- **Used By**: User scripts, integration packages, notebooks
