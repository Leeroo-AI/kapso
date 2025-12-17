# Implementation: set_adapter

> API Documentation for switching the active adapter in a multi-adapter PeftModel.

---

## Overview

| Property | Value |
|----------|-------|
| **Implementation Type** | API Doc |
| **Source File** | `src/peft/peft_model.py` |
| **Method** | `PeftModel.set_adapter` |
| **Paired Principle** | [[huggingface_peft_Adapter_Switching]] |
| **Parent Workflow** | [[huggingface_peft_Multi_Adapter_Management]] |

---

## Purpose

`set_adapter` changes which loaded adapter is currently active for forward passes. This enables instant behavior switching without reloading weights.

---

## API Signature

```python
model.set_adapter(
    adapter_name: str | list[str],
)
```

---

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `adapter_name` | `str \| list[str]` | Name(s) of adapter(s) to activate |

---

## Usage Example

### Basic Switching

```python
# Load multiple adapters
model = PeftModel.from_pretrained(base_model, "adapter-a")
model.load_adapter("adapter-b", "task_b")
model.load_adapter("adapter-c", "task_c")

# Switch to task_b
model.set_adapter("task_b")
print(model.active_adapter)  # "task_b"

# Generate with task_b behavior
output_b = model.generate(inputs)

# Switch to task_c
model.set_adapter("task_c")
output_c = model.generate(inputs)
```

### Multiple Active Adapters

```python
# Activate multiple adapters (for methods that support it)
model.set_adapter(["adapter_a", "adapter_b"])
```

---

## Key Behaviors

### Instant Switching

No weight loading occurs - just pointer changes:
- Zero latency switch
- No I/O operations
- No memory allocation

### Active Adapter State

```python
# Check current adapter
print(model.active_adapter)  # Single adapter name
print(model.active_adapters)  # List of active adapters
```

### Inference Mode Handling

Setting adapter also affects gradient computation:
```python
# For inference
model.set_adapter("task_a")
model.eval()

# For training
model.set_adapter("task_a")
model.train()
```

---

## Switching Patterns

### Pattern 1: Request-Based Routing

```python
def handle_request(model, request):
    task = classify_task(request)
    model.set_adapter(task)
    return model.generate(request.inputs)
```

### Pattern 2: Batch Processing

```python
# Process different tasks in batches
for task, batch in grouped_inputs.items():
    model.set_adapter(task)
    for input in batch:
        outputs.append(model.generate(input))
```

---

## Related Functions

| Function | Purpose |
|----------|---------|
| [[huggingface_peft_load_adapter]] | Load adapters before switching |
| [[huggingface_peft_delete_adapter]] | Remove adapters |
| `model.disable_adapters()` | Temporarily disable all adapters |

---

## Source Reference

- **File**: `src/peft/peft_model.py`
- **Method**: `set_adapter`

---

[[Category:Implementation]]
[[Category:huggingface_peft]]
[[Category:API Doc]]
