# Implementation: BitsAndBytesConfig

> Wrapper Documentation for configuring 4-bit/8-bit quantization for QLoRA.

---

## Overview

| Property | Value |
|----------|-------|
| **Implementation Type** | Wrapper Doc |
| **Source** | External (`transformers` library) |
| **Class** | `BitsAndBytesConfig` |
| **Paired Principle** | [[huggingface_peft_Quantization_Config]] |
| **Parent Workflow** | [[huggingface_peft_QLoRA_Training]] |

---

## Purpose

`BitsAndBytesConfig` configures quantization settings for loading models in reduced precision (4-bit or 8-bit). This is essential for QLoRA training, enabling fine-tuning of large models on limited GPU memory.

---

## API Signature

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    bnb_4bit_quant_type: str = "fp4",
    bnb_4bit_compute_dtype: torch.dtype = None,
    bnb_4bit_use_double_quant: bool = False,
)
```

---

## Key Parameters

| Parameter | Type | Recommended | Description |
|-----------|------|-------------|-------------|
| `load_in_4bit` | `bool` | `True` | Enable 4-bit quantization |
| `bnb_4bit_quant_type` | `str` | `"nf4"` | Quantization type (`"nf4"` or `"fp4"`) |
| `bnb_4bit_compute_dtype` | `dtype` | `torch.float16` | Dtype for compute operations |
| `bnb_4bit_use_double_quant` | `bool` | `True` | Enable double quantization |

---

## Usage for QLoRA

### Standard QLoRA Configuration

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
```

### Memory Comparison

| Configuration | 7B Model Memory |
|---------------|-----------------|
| FP32 | ~28 GB |
| FP16 | ~14 GB |
| 4-bit + double quant | ~4 GB |

---

## Integration with PEFT

After loading quantized model, apply LoRA normally:

```python
from peft import get_peft_model, LoraConfig

config = LoraConfig(
    r=16,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
```

---

## External Documentation

- **Official Docs**: [HuggingFace BitsAndBytes Integration](https://huggingface.co/docs/transformers/main/en/main_classes/quantization#bitsandbytes)

---

## Related Functions

| Function | Purpose |
|----------|---------|
| [[huggingface_peft_prepare_model_for_kbit_training]] | Prepare quantized model for training |
| [[huggingface_peft_get_peft_model]] | Apply LoRA to quantized model |

---

[[Category:Implementation]]
[[Category:huggingface_peft]]
[[Category:Wrapper Doc]]
