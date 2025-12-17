# Implementation: delete_adapter

> API Documentation for removing loaded adapters from a PeftModel.

---

## Overview

| Property | Value |
|----------|-------|
| **Implementation Type** | API Doc |
| **Source File** | `src/peft/peft_model.py` |
| **Method** | `PeftModel.delete_adapter` |
| **Paired Principle** | [[huggingface_peft_Adapter_Lifecycle]] |
| **Parent Workflow** | [[huggingface_peft_Multi_Adapter_Management]] |

---

## Purpose

`delete_adapter` removes a loaded adapter from memory, freeing up resources. This is useful for:
- Memory management with many adapters
- Cleaning up temporary/experimental adapters
- Lifecycle management in long-running services

---

## API Signature

```python
model.delete_adapter(
    adapter_name: str,
)
```

---

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `adapter_name` | `str` | Name of adapter to delete |

---

## Usage Example

### Basic Deletion

```python
# Load multiple adapters
model.load_adapter("adapter-a", "task_a")
model.load_adapter("adapter-b", "task_b")

# Delete one
model.delete_adapter("task_a")

# task_a is no longer available
print(list(model.peft_config.keys()))  # ["default", "task_b"]
```

### Memory Management

```python
# Process many adapters sequentially
for adapter_path in adapter_paths:
    model.load_adapter(adapter_path, "temp")
    process(model)
    model.delete_adapter("temp")  # Free memory
```

---

## Key Behaviors

### Active Adapter Handling

If deleting the active adapter:
- Another adapter becomes active
- If no adapters remain, model uses base weights

### Memory Release

Adapter weights are removed from GPU/CPU memory:
```python
# Before: GPU memory includes adapter weights
model.delete_adapter("large_adapter")
# After: Memory freed (may need torch.cuda.empty_cache())
```

---

## Related Functions

| Function | Purpose |
|----------|---------|
| [[huggingface_peft_load_adapter]] | Load adapters |
| [[huggingface_peft_set_adapter]] | Change active adapter |
| `model.disable_adapters()` | Temporarily disable without deleting |

---

## Source Reference

- **File**: `src/peft/peft_model.py`
- **Method**: `delete_adapter`

---

[[Category:Implementation]]
[[Category:huggingface_peft]]
[[Category:API Doc]]
