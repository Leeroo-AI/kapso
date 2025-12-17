# Implementation: get_peft_model

> API Documentation for converting a base model into a PEFT-enabled model with adapter injection.

---

## Overview

| Property | Value |
|----------|-------|
| **Implementation Type** | API Doc |
| **Source File** | `src/peft/mapping.py` + `src/peft/peft_model.py` |
| **Function** | `get_peft_model` |
| **Paired Principle** | [[huggingface_peft_PEFT_Application]] |
| **Parent Workflow** | [[huggingface_peft_LoRA_Finetuning]] |

---

## Purpose

`get_peft_model` is the primary entry point for converting a pretrained transformer model into a parameter-efficient fine-tuning (PEFT) model. It injects adapter layers (e.g., LoRA) into the specified target modules and returns a `PeftModel` wrapper.

---

## API Signature

```python
from peft import get_peft_model

peft_model = get_peft_model(
    model: PreTrainedModel,
    peft_config: PeftConfig,
    adapter_name: str = "default",
    mixed: bool = False,
    autocast_adapter_dtype: bool = True,
    revision: str = None,
    low_cpu_mem_usage: bool = False,
)
```

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `PreTrainedModel` | required | The base model to add adapters to |
| `peft_config` | `PeftConfig` | required | Configuration specifying adapter type and settings |
| `adapter_name` | `str` | `"default"` | Name identifier for this adapter |
| `mixed` | `bool` | `False` | Enable mixed adapter types on same model |
| `autocast_adapter_dtype` | `bool` | `True` | Auto-cast adapter weights to model dtype |
| `low_cpu_mem_usage` | `bool` | `False` | Initialize adapters on meta device for memory efficiency |

---

## Usage Example

### Basic Usage

```python
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
)

# Configure LoRA
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)

# Create PEFT model
model = get_peft_model(base_model, config)

# Check trainable parameters
model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622
```

### Named Adapter

```python
model = get_peft_model(
    base_model,
    config,
    adapter_name="custom_adapter"
)
```

---

## Return Value

Returns a `PeftModel` instance that wraps the original model with adapter layers injected.

| Property | Type | Description |
|----------|------|-------------|
| `PeftModel.base_model` | `BaseTuner` | The model with adapters injected |
| `PeftModel.peft_config` | `dict[str, PeftConfig]` | Mapping of adapter names to configs |
| `PeftModel.active_adapter` | `str` | Currently active adapter name |

---

## Key Behaviors

### Model Mutation

`get_peft_model` modifies the original model in-place before wrapping it:

1. **Injects adapter layers**: Replaces target modules with adapter-wrapped versions
2. **Freezes base weights**: Sets `requires_grad=False` on base model parameters
3. **Enables adapter weights**: Sets `requires_grad=True` on adapter parameters

### Alternative: inject_adapter_in_model

For cases where you need the mutated model without the `PeftModel` wrapper:

```python
from peft import inject_adapter_in_model

# Returns mutated model directly (not wrapped)
model = inject_adapter_in_model(peft_config, model, adapter_name="default")
```

---

## Related Functions

| Function | Purpose |
|----------|---------|
| [[huggingface_peft_LoraConfig]] | Configuration for LoRA adapters |
| [[huggingface_peft_save_pretrained]] | Save trained adapters |
| [[huggingface_peft_PeftModel_from_pretrained]] | Load adapters from disk |

---

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: Target modules not found` | Invalid `target_modules` specification | Check module names with `model.named_modules()` |
| `ValueError: Unknown PEFT type` | Invalid `peft_config` type | Use valid config class (LoraConfig, etc.) |

---

## Source Reference

- **Mapping Registry**: `src/peft/mapping.py`
- **PeftModel Creation**: `src/peft/peft_model.py`

---

[[Category:Implementation]]
[[Category:huggingface_peft]]
[[Category:API Doc]]
