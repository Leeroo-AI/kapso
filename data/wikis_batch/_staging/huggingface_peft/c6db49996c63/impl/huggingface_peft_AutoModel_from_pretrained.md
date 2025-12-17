# Implementation: AutoModel_from_pretrained

> Wrapper Documentation for loading pretrained transformer models from HuggingFace.

---

## Overview

| Property | Value |
|----------|-------|
| **Implementation Type** | Wrapper Doc |
| **Source** | External (`transformers` library) |
| **Function** | `AutoModelForCausalLM.from_pretrained` |
| **Paired Principle** | [[huggingface_peft_Model_Loading]] |
| **Parent Workflow** | [[huggingface_peft_LoRA_Finetuning]] |

---

## Purpose

This is the standard HuggingFace Transformers method for loading pretrained models. In the context of PEFT, it loads the base model that will receive adapter injection.

---

## API Signature

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path: str,
    device_map: str | dict = None,
    torch_dtype: torch.dtype = None,
    quantization_config: BitsAndBytesConfig = None,
    attn_implementation: str = None,
    **kwargs,
)
```

---

## Key Parameters for PEFT Workflows

| Parameter | Type | Recommended | Description |
|-----------|------|-------------|-------------|
| `pretrained_model_name_or_path` | `str` | required | HuggingFace model ID or local path |
| `device_map` | `str \| dict` | `"auto"` | Automatic device placement for large models |
| `torch_dtype` | `torch.dtype` | `torch.float16` | Model precision (fp16/bf16 for efficiency) |
| `trust_remote_code` | `bool` | `True` | Required for custom model architectures |

---

## Usage in PEFT Context

### Standard Loading

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",
    torch_dtype=torch.float16,
)
```

### Loading for QLoRA

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
```

---

## Integration with PEFT

After loading, pass the model to `get_peft_model`:

```python
from peft import get_peft_model, LoraConfig

config = LoraConfig(r=16, target_modules="all-linear")
peft_model = get_peft_model(model, config)
```

---

## External Documentation

- **Official Docs**: [HuggingFace Transformers - Loading Models](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained)

---

## Related Functions

| Function | Purpose |
|----------|---------|
| [[huggingface_peft_get_peft_model]] | Convert loaded model to PEFT model |
| [[huggingface_peft_BitsAndBytesConfig]] | Quantization configuration |

---

[[Category:Implementation]]
[[Category:huggingface_peft]]
[[Category:Wrapper Doc]]
