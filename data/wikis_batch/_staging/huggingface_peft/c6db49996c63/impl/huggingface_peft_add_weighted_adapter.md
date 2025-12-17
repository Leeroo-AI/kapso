# Implementation: add_weighted_adapter

> API Documentation for combining multiple adapters with weighted averaging.

---

## Overview

| Property | Value |
|----------|-------|
| **Implementation Type** | API Doc |
| **Source File** | `src/peft/tuners/lora/model.py` |
| **Method** | `LoraModel.add_weighted_adapter` |
| **Paired Principle** | [[huggingface_peft_Adapter_Combination]] |
| **Parent Workflow** | [[huggingface_peft_Multi_Adapter_Management]] |

---

## Purpose

`add_weighted_adapter` creates a new adapter by combining multiple existing adapters with specified weights. This enables:
- Task interpolation between specialized adapters
- Ensemble-like behavior from single models
- Custom adapter blending

---

## API Signature

```python
model.add_weighted_adapter(
    adapters: list[str],
    weights: list[float],
    adapter_name: str,
    combination_type: str = "linear",
)
```

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adapters` | `list[str]` | required | Names of adapters to combine |
| `weights` | `list[float]` | required | Weights for each adapter |
| `adapter_name` | `str` | required | Name for the new combined adapter |
| `combination_type` | `str` | `"linear"` | Combination method |

---

## Usage Example

### Basic Adapter Combination

```python
# Load multiple adapters
model = PeftModel.from_pretrained(base_model, "adapter-a")
model.load_adapter("adapter-b", "task_b")
model.load_adapter("adapter-c", "task_c")

# Create combined adapter
model.add_weighted_adapter(
    adapters=["default", "task_b", "task_c"],
    weights=[0.5, 0.3, 0.2],
    adapter_name="combined",
)

# Use combined adapter
model.set_adapter("combined")
```

### Task Interpolation

```python
# Interpolate between two tasks
model.add_weighted_adapter(
    adapters=["summarization", "translation"],
    weights=[0.7, 0.3],
    adapter_name="sum_trans_blend",
)
```

---

## Combination Types

| Type | Formula | Use Case |
|------|---------|----------|
| `"linear"` | `sum(w_i * A_i)` | Standard weighted average |
| `"cat"` | Concatenation-based | Preserving individual capacities |
| `"svd"` | SVD-based merge | Rank reduction |
| `"ties"` | TIES merging | Reducing interference |
| `"dare_ties"` | DARE + TIES | Advanced merging |

---

## Key Behaviors

### Weight Normalization

Weights are typically used as-is. For normalized blending:
```python
weights = [w / sum(weights) for w in raw_weights]
```

### New Adapter Creation

A new adapter is created; original adapters remain unchanged:
```python
# Before: adapters = ["a", "b"]
model.add_weighted_adapter(["a", "b"], [0.5, 0.5], "combined")
# After: adapters = ["a", "b", "combined"]
```

---

## Related Functions

| Function | Purpose |
|----------|---------|
| [[huggingface_peft_load_adapter]] | Load adapters to combine |
| [[huggingface_peft_set_adapter]] | Activate combined adapter |
| [[huggingface_peft_delete_adapter]] | Remove adapters |

---

## Source Reference

- **File**: `src/peft/tuners/lora/model.py`
- **Method**: `add_weighted_adapter`

---

[[Category:Implementation]]
[[Category:huggingface_peft]]
[[Category:API Doc]]
