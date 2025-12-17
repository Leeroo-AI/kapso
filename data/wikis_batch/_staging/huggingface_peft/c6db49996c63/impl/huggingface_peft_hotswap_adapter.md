# Implementation: hotswap_adapter

> API Documentation for replacing adapter weights without reloading the base model.

---

## Overview

| Property | Value |
|----------|-------|
| **Implementation Type** | API Doc |
| **Source File** | `src/peft/utils/hotswap.py:L545-631` |
| **Function** | `hotswap_adapter` |
| **Paired Principle** | [[huggingface_peft_Hotswap_Execution]] |
| **Parent Workflow** | [[huggingface_peft_Adapter_Hotswapping]] |

---

## Purpose

`hotswap_adapter` substitutes adapter weights in-place, keeping the base model and PEFT structure intact. This enables:
- Zero-downtime adapter updates in production
- Fast adapter switching with compiled models
- Memory-efficient adapter replacement

---

## API Signature

```python
from peft.utils.hotswap import hotswap_adapter

hotswap_adapter(
    model: PeftModel,
    model_name_or_path: str,
    adapter_name: str,
    torch_device: str | None = None,
    **kwargs,
)
```

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `PeftModel` | required | Model with adapter to replace |
| `model_name_or_path` | `str` | required | Path to new adapter weights |
| `adapter_name` | `str` | required | Name of adapter to replace (e.g., `"default"`) |
| `torch_device` | `str` | `None` | Device for new weights |
| `**kwargs` | | | HuggingFace Hub kwargs (token, revision, etc.) |

---

## Usage Example

### Basic Hot-Swap

```python
from peft import PeftModel
from peft.utils.hotswap import hotswap_adapter

# Load model with initial adapter
model = PeftModel.from_pretrained(base_model, "adapter-v1")

# Use adapter-v1 for inference
outputs_v1 = model.generate(inputs)

# Hot-swap to v2 (same adapter name, new weights)
hotswap_adapter(model, "adapter-v2", "default", torch_device="cuda")

# Now using adapter-v2
outputs_v2 = model.generate(inputs)
```

### With Compiled Model

```python
from peft.utils.hotswap import prepare_model_for_compiled_hotswap, hotswap_adapter

model = PeftModel.from_pretrained(base_model, "adapter-r16")

# Prepare and compile
prepare_model_for_compiled_hotswap(model, target_rank=32)
model = torch.compile(model)

# Hot-swap works without recompilation
hotswap_adapter(model, "adapter-r8", "default")  # Different rank OK
```

---

## Key Behaviors

### Weight Replacement

The function:
1. Loads new adapter config and validates compatibility
2. Loads new weights to specified device
3. Copies weights into existing tensors (in-place)
4. Updates scaling factors

### Compatibility Checks

Before swapping, validates:
- Same PEFT type (e.g., both LoRA)
- Compatible config parameters (dropout, use_dora, etc.)
- Target modules are subset of existing

### Handling Rank Mismatches

| Scenario | Behavior |
|----------|----------|
| Same rank | Direct weight copy |
| New adapter smaller | Zero-fill unused dimensions |
| New adapter larger | Error (unless prepared with target_rank) |

---

## Production Patterns

### Pattern 1: Version Updates

```python
# Zero-downtime adapter update
def update_adapter(model, new_version):
    hotswap_adapter(
        model,
        f"s3://adapters/{new_version}",
        "default",
    )
    log(f"Updated to {new_version}")
```

### Pattern 2: A/B Testing

```python
import random

def inference_with_ab_test(model, inputs, adapters):
    variant = random.choice(["A", "B"])
    hotswap_adapter(model, adapters[variant], "default")
    return model.generate(inputs), variant
```

---

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| `RuntimeError: unexpected keys` | Adapter targets different layers | Use compatible adapter |
| `ValueError: Incompatible configs` | Different dropout/dora settings | Match config parameters |
| `ValueError: Incompatible shapes` | Rank too large | Use `prepare_model_for_compiled_hotswap` |

---

## Related Functions

| Function | Purpose |
|----------|---------|
| [[huggingface_peft_prepare_model_for_compiled_hotswap]] | Prepare for rank mismatches |
| `hotswap_adapter_from_state_dict` | Low-level swap from dict |

---

## Source Reference

- **File**: `src/peft/utils/hotswap.py`
- **Lines**: 545-631
- **Function**: `hotswap_adapter`

---

[[Category:Implementation]]
[[Category:huggingface_peft]]
[[Category:API Doc]]
