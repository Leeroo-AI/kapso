# Implementation: load_adapter

> API Documentation for loading additional adapters onto an existing PeftModel.

---

## Overview

| Property | Value |
|----------|-------|
| **Implementation Type** | API Doc |
| **Source File** | `src/peft/peft_model.py` |
| **Method** | `PeftModel.load_adapter` |
| **Paired Principle** | [[huggingface_peft_Adapter_Addition]] |
| **Parent Workflow** | [[huggingface_peft_Multi_Adapter_Management]] |

---

## Purpose

`load_adapter` adds a new adapter to an existing PeftModel without replacing the current adapters. This enables multi-adapter scenarios where different adapters can be loaded and switched between.

---

## API Signature

```python
model.load_adapter(
    model_id: str | os.PathLike,
    adapter_name: str,
    is_trainable: bool = False,
    torch_device: str | None = None,
    autocast_adapter_dtype: bool = True,
    **kwargs,
)
```

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | `str \| PathLike` | required | HuggingFace Hub ID or local path |
| `adapter_name` | `str` | required | Unique name for this adapter |
| `is_trainable` | `bool` | `False` | Enable gradients for training |
| `torch_device` | `str` | `None` | Device to load adapter onto |

---

## Usage Example

### Loading Multiple Adapters

```python
from peft import PeftModel

# Start with one adapter
model = PeftModel.from_pretrained(base_model, "adapter-a")

# Load additional adapters
model.load_adapter("adapter-b", adapter_name="task_b")
model.load_adapter("adapter-c", adapter_name="task_c")

# List all adapters
print(list(model.peft_config.keys()))
# ['default', 'task_b', 'task_c']
```

### Switch Between Adapters

```python
# Use task_b adapter
model.set_adapter("task_b")
outputs_b = model.generate(inputs)

# Switch to task_c
model.set_adapter("task_c")
outputs_c = model.generate(inputs)
```

---

## Key Behaviors

### Adapter Name Requirements

- Must be unique across loaded adapters
- Cannot be `"default"` if default adapter exists
- Used to reference adapter in `set_adapter()`

### Architecture Compatibility

All loaded adapters must:
- Target the same or subset of modules
- Be compatible with the base model
- Have compatible PEFT types (all LoRA, etc.)

### Memory Impact

Each loaded adapter adds:
- LoRA A and B matrices per targeted layer
- Approximately `(d_in + d_out) * r` parameters per layer

---

## Multi-Adapter Patterns

### Pattern 1: Task-Specific Adapters

```python
model.load_adapter("./adapters/summarization", "summarize")
model.load_adapter("./adapters/translation", "translate")
model.load_adapter("./adapters/qa", "question_answer")

# Route based on task
def run_task(model, task, inputs):
    model.set_adapter(task)
    return model.generate(inputs)
```

### Pattern 2: Language Adapters

```python
model.load_adapter("./adapters/english", "en")
model.load_adapter("./adapters/french", "fr")
model.load_adapter("./adapters/german", "de")

# Switch based on detected language
model.set_adapter(detected_language)
```

---

## Related Functions

| Function | Purpose |
|----------|---------|
| [[huggingface_peft_set_adapter]] | Switch active adapter |
| [[huggingface_peft_delete_adapter]] | Remove loaded adapter |
| [[huggingface_peft_add_weighted_adapter]] | Combine adapters |

---

## Source Reference

- **File**: `src/peft/peft_model.py`
- **Method**: `load_adapter`

---

[[Category:Implementation]]
[[Category:huggingface_peft]]
[[Category:API Doc]]
