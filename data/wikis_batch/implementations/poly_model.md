# Poly Model Implementation

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/tuners/poly/model.py`
- **Lines**: 104
- **Purpose**: Poly model wrapper with task ID injection via hooks

## Overview

PolyModel manages Poly adapter application and implements a hook system to inject task_ids into forward passes. This is necessary because Poly layers require task IDs to determine skill routing, but standard model interfaces don't include this parameter.

## PolyModel Class

**Inheritance**: Extends `BaseTuner`

**Class Attributes**:
- `prefix`: `"poly_"`
- `tuner_layer_cls`: `PolyLayer`
- `target_module_mapping`: `TRANSFORMERS_MODELS_TO_POLY_TARGET_MODULES_MAPPING`

## Core Methods

### 1. `_create_and_replace`

Creates Poly layer and replaces target module.

**Implementation**:
```python
def _create_and_replace(
    self,
    poly_config: PolyConfig,
    adapter_name: str,
    target: nn.Module,
    target_name: str,
    parent: nn.Module,
    **optional_kwargs,
):
    if isinstance(target, PolyLayer):
        # Update existing Poly layer
        target.update_layer(adapter_name, poly_config)
    else:
        # Create new Poly layer
        new_module = self._create_new_module(
            poly_config,
            adapter_name,
            target,
        )
        if adapter_name not in self.active_adapters:
            new_module.requires_grad_(False)
        self._replace_module(parent, target_name, new_module, target)
```

**Simple Design**: No complex configuration needed, just pass poly_config

### 2. `_create_new_module` (static)

Factory method for creating Poly layers.

**Implementation**:
```python
@staticmethod
def _create_new_module(poly_config, adapter_name, target, **kwargs):
    # Extract base layer
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    # Validate and create
    if isinstance(target_base_layer, torch.nn.Linear):
        return Linear(target, adapter_name, poly_config, **kwargs)
    else:
        raise ValueError(
            f"Target module {target} is not supported. "
            "Currently, only the following modules are supported: `torch.nn.Linear`."
        )
```

**Supported**: Only `torch.nn.Linear`

## Task ID Injection System

The core innovation of PolyModel is automatic task_id injection.

### Problem

Poly layers need task_ids:
```python
class Linear(nn.Module):
    def forward(self, x, *args, task_ids=None, **kwargs):
        # Need task_ids to route skills
        mixing_weights = poly_router(task_ids=task_ids, ...)
```

But standard models don't pass task_ids:
```python
# Standard model call
output = model(input_ids)  # No task_ids parameter!
```

### Solution: Pre-Hooks

PolyModel uses PyTorch forward pre-hooks to inject task_ids:

```python
def _register_pre_hooks(self, task_ids):
    """Register pre hooks to inject task_ids."""
    if task_ids is None:
        return []

    def pre_hook(_, args, kwargs):
        # Inject task_ids into kwargs
        kwargs["task_ids"] = task_ids
        return args, kwargs

    handles = []
    for module in self.model.modules():
        if isinstance(module, Linear):
            handle = module.register_forward_pre_hook(pre_hook, with_kwargs=True)
            handles.append(handle)

    return handles
```

**How It Works**:
1. User calls `model.forward(x, task_ids=task_ids)`
2. PolyModel registers pre-hooks on all Poly Linear layers
3. Before each Linear forward, hook adds task_ids to kwargs
4. Linear receives task_ids automatically
5. After forward, hooks are removed

### Context Manager

```python
@contextmanager
def _manage_pre_hooks(self, task_ids):
    """Context manager to handle lifecycle of pre hooks."""
    handles = self._register_pre_hooks(task_ids)
    try:
        yield
    finally:
        # Always remove hooks
        for handle in handles:
            handle.remove()
```

**Benefits**:
- Automatic cleanup
- Exception-safe
- No lingering hooks

## Custom Forward and Generate

### Forward Method

```python
def forward(self, *args, task_ids=None, **kwargs):
    with self._manage_pre_hooks(task_ids):
        return self.model(*args, **kwargs)
```

**Usage**:
```python
# Training
output = model(input_ids, task_ids=task_ids)

# Internally:
# 1. Registers hooks with task_ids
# 2. Calls base model forward
# 3. Hooks inject task_ids into Poly layers
# 4. Hooks removed after forward
```

### Generate Method

```python
def generate(self, *args, task_ids=None, **kwargs):
    with self._manage_pre_hooks(task_ids):
        return self.model.generate(*args, **kwargs)
```

**Usage**:
```python
# Generation
output = model.generate(input_ids, task_ids=task_ids, max_length=50)

# Same hook mechanism applies during generation
```

## Application Workflow

```
1. User creates PolyConfig

2. get_peft_model(base_model, poly_config)
   └─> Creates PolyModel

3. PolyModel.__init__() → inject_adapter()
   └─> For each target module:
       └─> _create_and_replace()
           └─> Creates Linear(target, adapter_name, poly_config)

4. Model ready, no hooks registered yet

5. User calls model.forward(x, task_ids=task_ids)
   └─> PolyModel.forward()
       ├─> _manage_pre_hooks(task_ids) enters
       │   └─> Registers pre-hooks on all Poly Linear layers
       ├─> Calls self.model.forward(x)
       │   └─> For each Poly Linear:
       │       ├─> Pre-hook fires, injects task_ids into kwargs
       │       └─> Linear.forward(x, task_ids=task_ids)
       │           └─> Router uses task_ids
       └─> _manage_pre_hooks exits, removes all hooks

6. Next forward pass repeats hook registration/removal
```

## Design Patterns

### 1. Hook-Based Parameter Injection
Elegant solution to parameter passing problem:
- No model interface changes
- Automatic propagation
- Clean separation of concerns

### 2. Context Manager Pattern
Safe resource management:
```python
with self._manage_pre_hooks(task_ids):
    # Hooks active
    result = computation()
    # Hooks automatically cleaned up
```

### 3. Module Filtering
Only hooks Poly layers:
```python
for module in self.model.modules():
    if isinstance(module, Linear):  # Only Poly Linear
        handle = module.register_forward_pre_hook(...)
```

### 4. Simple Factory
Straightforward layer creation:
```python
if isinstance(target_base_layer, torch.nn.Linear):
    return Linear(target, adapter_name, poly_config)
```

## Example Usage

### Basic Training
```python
from transformers import AutoModelForCausalLM
from peft import PolyConfig, get_peft_model
import torch

base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

config = PolyConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    n_tasks=10,
    n_skills=4,
    n_splits=1
)

model = get_peft_model(base_model, config)

# Training loop
for batch in dataloader:
    input_ids = batch["input_ids"]
    task_ids = batch["task_ids"]  # Shape: (batch_size,)

    # Forward pass - task_ids injected via hooks
    output = model(input_ids, task_ids=task_ids)

    loss = compute_loss(output, batch["labels"])
    loss.backward()
    optimizer.step()
```

### Generation
```python
# Generate with specific task
task_id = torch.tensor([3])  # Task 3

output = model.generate(
    input_ids,
    task_ids=task_id,
    max_length=50,
    num_beams=5
)
```

### Multi-Task Batch
```python
# Batch with different tasks
input_ids = torch.tensor([...])  # Shape: (4, seq_len)
task_ids = torch.tensor([0, 1, 2, 0])  # Different tasks in batch

# Each sample routes skills according to its task_id
output = model(input_ids, task_ids=task_ids)
```

### Inference Without Task ID
```python
# If task_ids=None, router must handle it
# (Current implementation expects task_ids)
output = model(input_ids, task_ids=None)
# May fail or use default routing depending on router implementation
```

## Comparison with Other PEFT Methods

### Hook Usage

**Poly**:
- Uses hooks for task_id injection
- Dynamic hook registration/removal
- Per-forward overhead (minimal)

**LoRA/VeRA/VBLoRA/SHiRA/C3A**:
- No hooks needed
- Direct parameter passing
- Slightly lower overhead

### Model Interface

**Poly**:
```python
output = model(input_ids, task_ids=task_ids)  # Extra parameter
```

**Others**:
```python
output = model(input_ids)  # Standard interface
```

### Complexity

**Poly**: Most complex (hooks + routing + multi-skill)
**Others**: Simpler (direct adaptation)

## Limitations

1. **Layer Support**: Only nn.Linear
2. **Task ID Required**: Must provide task_ids (or handle None)
3. **Hook Overhead**: Small but present
4. **Complexity**: More complex than standard LoRA

## Implementation Notes

1. **Hook Registration**: Per forward pass
2. **Hook Cleanup**: Automatic via context manager
3. **Exception Safety**: Hooks removed even if forward fails
4. **Module Filtering**: Only affects Poly Linear layers
5. **No Persistent Hooks**: Clean slate each forward

## Future Enhancements

Potential improvements:

1. **Default Task Routing**:
   ```python
   if task_ids is None:
       # Use input-based routing
       task_ids = input_based_router(x)
   ```

2. **Persistent Hooks**:
   ```python
   # For inference mode, keep hooks registered
   if inference_mode:
       self.persistent_hooks = self._register_pre_hooks(default_task_id)
   ```

3. **Quantization Support**:
   ```python
   if loaded_in_8bit:
       return Linear8bitLt(target, adapter_name, poly_config)
   ```

4. **Conv Layer Support**:
   ```python
   elif isinstance(target_base_layer, nn.Conv2d):
       return Conv2d(target, adapter_name, poly_config)
   ```

## References

- **Polytropon**: https://huggingface.co/papers/2202.13914
- **Multi-Head Routing**: https://huggingface.co/papers/2211.03831
- **Hook System**: Elegant solution for parameter injection in multi-task scenarios
- **Design**: Clean separation between routing logic and parameter passing
