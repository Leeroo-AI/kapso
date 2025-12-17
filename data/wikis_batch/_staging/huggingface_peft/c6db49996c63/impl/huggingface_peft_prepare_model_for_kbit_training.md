# Implementation: prepare_model_for_kbit_training

> API Documentation for preparing quantized models for gradient computation.

---

## Overview

| Property | Value |
|----------|-------|
| **Implementation Type** | API Doc |
| **Source File** | `src/peft/utils/other.py:L130-215` |
| **Function** | `prepare_model_for_kbit_training` |
| **Paired Principle** | [[huggingface_peft_Memory_Optimization]] |
| **Parent Workflow** | [[huggingface_peft_QLoRA_Training]] |

---

## Purpose

`prepare_model_for_kbit_training` prepares a quantized (4-bit or 8-bit) model for training by:
- Freezing base model parameters
- Enabling gradient checkpointing
- Casting layer norms to float32 for stability
- Setting up input embeddings for gradient flow

---

## API Signature

```python
from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(
    model: PreTrainedModel,
    use_gradient_checkpointing: bool = True,
    gradient_checkpointing_kwargs: dict | None = None,
)
```

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `PreTrainedModel` | required | Quantized model to prepare |
| `use_gradient_checkpointing` | `bool` | `True` | Enable gradient checkpointing |
| `gradient_checkpointing_kwargs` | `dict` | `None` | Additional checkpointing options |

---

## Usage Example

### QLoRA Training Setup

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig

# Load 4-bit model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
)

# Prepare for training
model = prepare_model_for_kbit_training(model)

# Apply LoRA
config = LoraConfig(r=16, target_modules="all-linear")
model = get_peft_model(model, config)
```

---

## Key Behaviors

### 1. Parameter Freezing

```python
for param in model.parameters():
    param.requires_grad = False
```

### 2. Layer Norm Casting

Non-INT8 parameters are cast to float32 for numerical stability:
```python
if param.dtype in (torch.float16, torch.bfloat16):
    param.data = param.data.to(torch.float32)
```

### 3. Gradient Checkpointing

Enables `use_reentrant` gradient checkpointing with input gradient hooks:
```python
model.gradient_checkpointing_enable(**gradient_checkpointing_kwargs)
```

---

## Memory Impact

| Feature | Memory Saved | Trade-off |
|---------|--------------|-----------|
| Gradient checkpointing | ~40-60% | Slower backward pass |
| 4-bit quantization | ~75% | Slight quality loss |
| Combined | ~85% | Training possible on consumer GPUs |

---

## When to Use

| Scenario | Use This Function |
|----------|-------------------|
| QLoRA training | Yes |
| Standard LoRA | Not needed |
| GPTQ models | Yes |
| Full precision models | Not needed |

---

## Related Functions

| Function | Purpose |
|----------|---------|
| [[huggingface_peft_BitsAndBytesConfig]] | Configure quantization |
| [[huggingface_peft_get_peft_model]] | Apply LoRA after preparation |

---

## Source Reference

- **File**: `src/peft/utils/other.py`
- **Lines**: 130-215
- **Function**: `prepare_model_for_kbit_training`

---

[[Category:Implementation]]
[[Category:huggingface_peft]]
[[Category:API Doc]]
