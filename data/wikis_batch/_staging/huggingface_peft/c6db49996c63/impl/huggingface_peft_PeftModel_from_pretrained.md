# Implementation: PeftModel_from_pretrained

> API Documentation for loading trained PEFT adapters onto a base model.

---

## Overview

| Property | Value |
|----------|-------|
| **Implementation Type** | API Doc |
| **Source File** | `src/peft/peft_model.py:L389-700` |
| **Method** | `PeftModel.from_pretrained` |
| **Paired Principle** | [[huggingface_peft_Adapter_Loading]] |
| **Parent Workflow** | [[huggingface_peft_Adapter_Inference]], [[huggingface_peft_Adapter_Hotswapping]] |

---

## Purpose

`PeftModel.from_pretrained` loads a saved PEFT adapter and attaches it to a base model, creating a ready-to-use `PeftModel` for inference or continued training.

---

## API Signature

```python
from peft import PeftModel

peft_model = PeftModel.from_pretrained(
    model: torch.nn.Module,
    model_id: str | os.PathLike,
    adapter_name: str = "default",
    is_trainable: bool = False,
    config: PeftConfig | None = None,
    revision: str | None = None,
    torch_device: str | None = None,
    autocast_adapter_dtype: bool = True,
    **kwargs,
)
```

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | required | Base model to attach adapter to |
| `model_id` | `str \| PathLike` | required | HuggingFace Hub ID or local path to adapter |
| `adapter_name` | `str` | `"default"` | Name for this adapter instance |
| `is_trainable` | `bool` | `False` | Set `True` for continued training |
| `config` | `PeftConfig` | `None` | Override saved config |
| `torch_device` | `str` | `None` | Device to load adapter onto |

---

## Usage Example

### Basic Inference Loading

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",
    torch_dtype=torch.float16,
)

# Load adapter
model = PeftModel.from_pretrained(
    base_model,
    "username/my-lora-adapter",
)

# Ready for inference
model.eval()
```

### Loading for Continued Training

```python
model = PeftModel.from_pretrained(
    base_model,
    "./checkpoint-1000",
    is_trainable=True,  # Enable gradients
)
```

### Loading with Custom Name

```python
model = PeftModel.from_pretrained(
    base_model,
    "adapter-path",
    adapter_name="task_specific",
)
print(model.active_adapter)  # "task_specific"
```

---

## Return Value

Returns a `PeftModel` instance with:

| Property | Type | Description |
|----------|------|-------------|
| `base_model` | `BaseTuner` | Model with adapters injected |
| `peft_config` | `dict` | Mapping of adapter names to configs |
| `active_adapter` | `str` | Currently active adapter |

---

## Key Behaviors

### Automatic Configuration Loading

The method reads `adapter_config.json` from the specified path to reconstruct the adapter architecture.

### Device Handling

If `torch_device` is not specified:
1. Uses model's device if consistent
2. Falls back to first available GPU
3. Uses CPU if no GPU available

### Base Model Compatibility

The base model must match the one used during training:
- Same architecture
- Same hidden dimensions
- Compatible tokenizer

---

## Loading from HuggingFace Hub

```python
# Public adapter
model = PeftModel.from_pretrained(base_model, "username/adapter-name")

# Private adapter with authentication
model = PeftModel.from_pretrained(
    base_model,
    "username/private-adapter",
    token="hf_...",
)
```

---

## Related Functions

| Function | Purpose |
|----------|---------|
| [[huggingface_peft_save_pretrained]] | Save adapters for later loading |
| [[huggingface_peft_load_adapter]] | Load additional adapters |
| [[huggingface_peft_merge_and_unload]] | Merge adapter into base model |

---

## Source Reference

- **File**: `src/peft/peft_model.py`
- **Lines**: 389-700
- **Method**: `classmethod from_pretrained`

---

[[Category:Implementation]]
[[Category:huggingface_peft]]
[[Category:API Doc]]
