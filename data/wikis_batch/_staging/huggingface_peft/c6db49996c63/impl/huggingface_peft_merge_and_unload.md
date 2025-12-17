# Implementation: merge_and_unload

> API Documentation for permanently merging adapter weights into the base model.

---

## Overview

| Property | Value |
|----------|-------|
| **Implementation Type** | API Doc |
| **Source File** | `src/peft/peft_model.py` (delegates to tuner) |
| **Method** | `PeftModel.merge_and_unload` |
| **Paired Principle** | [[huggingface_peft_Adapter_Merging]] |
| **Parent Workflow** | [[huggingface_peft_Adapter_Inference]] |

---

## Purpose

`merge_and_unload` permanently fuses adapter weights into the base model, returning a standard `PreTrainedModel` without any PEFT wrapper. This is useful for:
- Eliminating adapter overhead during inference
- Exporting to formats that don't support adapters
- Creating standalone fine-tuned models

---

## API Signature

```python
merged_model = model.merge_and_unload(
    progressbar: bool = False,
    safe_merge: bool = False,
    adapter_names: list[str] | None = None,
)
```

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `progressbar` | `bool` | `False` | Show progress during merge |
| `safe_merge` | `bool` | `False` | Check for NaN/Inf after merge |
| `adapter_names` | `list[str] \| None` | `None` | Specific adapters to merge; `None` = active adapter |

---

## Usage Example

### Basic Merge

```python
from peft import PeftModel

# Load model with adapter
model = PeftModel.from_pretrained(base_model, "adapter-path")

# Merge and get standard model
merged_model = model.merge_and_unload()

# merged_model is now a standard PreTrainedModel
print(type(merged_model))  # transformers.models.llama.modeling_llama.LlamaForCausalLM
```

### Safe Merge with Progress

```python
merged_model = model.merge_and_unload(
    progressbar=True,
    safe_merge=True,  # Check for numerical issues
)
```

### Merge Specific Adapters

```python
# With multiple adapters loaded
merged_model = model.merge_and_unload(
    adapter_names=["task_a", "task_b"],
)
```

---

## Return Value

Returns a standard `PreTrainedModel` with adapter weights permanently merged into base weights.

---

## Key Behaviors

### What Happens During Merge

For each LoRA layer:
```python
# Conceptually:
W_merged = W_base + (lora_A @ lora_B) * scaling
```

The low-rank decomposition is computed and added to the base weights.

### Irreversible Operation

After merging:
- Original base weights are modified
- Adapter layers are removed
- Cannot switch adapters
- Cannot unmerge

### Memory Impact

| Before Merge | After Merge |
|--------------|-------------|
| Base + Adapter weights | Merged weights only |
| PeftModel wrapper | Standard model |
| Can switch adapters | Single configuration |

---

## When to Use

| Scenario | Recommendation |
|----------|----------------|
| Production deployment | Merge for speed |
| Multi-adapter serving | Keep separate |
| Export to ONNX/TensorRT | Merge first |
| Continued training | Don't merge |

---

## Alternative: merge() Without Unload

To merge but keep PEFT wrapper:

```python
# Merge but keep PeftModel structure
model.merge_adapter()

# Can still unmerge later
model.unmerge_adapter()
```

---

## Related Functions

| Function | Purpose |
|----------|---------|
| [[huggingface_peft_PeftModel_from_pretrained]] | Load adapter before merge |
| `model.merge_adapter()` | Merge without removing wrapper |
| `model.unmerge_adapter()` | Reverse in-place merge |

---

## Source Reference

- **File**: `src/peft/peft_model.py`
- **Method**: `merge_and_unload`
- **Delegates to**: `BaseTuner.merge_and_unload`

---

[[Category:Implementation]]
[[Category:huggingface_peft]]
[[Category:API Doc]]
