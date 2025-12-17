# Implementation: huggingface_peft_AutoModel_from_pretrained

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers AutoModel|https://huggingface.co/docs/transformers/model_doc/auto]]
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Model_Loading]], [[domain::Transformers]], [[domain::External]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

| Property | Value |
|----------|-------|
| **Type** | Wrapper Doc |
| **Source** | External (transformers library) |
| **Principle** | [[implemented_by::Implementation:huggingface_peft_AutoModel_from_pretrained]] |
| **Environment** | [[requires_env::Environment:huggingface_peft_CUDA_Training]] |

== Description ==

`AutoModelForCausalLM.from_pretrained()` (and related Auto classes) is the transformers library function for loading pretrained models. This is the standard entry point for loading base models before applying PEFT adapters.

== API Signature ==

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path: str,
    device_map: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    quantization_config: Optional[BitsAndBytesConfig] = None,
    trust_remote_code: bool = False,
    **kwargs
)
```

== Parameters ==

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pretrained_model_name_or_path` | str | required | HF Hub ID or local path |
| `device_map` | str | None | Device placement ("auto", "cuda:0", etc.) |
| `torch_dtype` | torch.dtype | None | Model precision (float16, bfloat16) |
| `quantization_config` | BitsAndBytesConfig | None | Quantization settings |
| `trust_remote_code` | bool | False | Allow custom model code |

== Returns ==

`PreTrainedModel` - The loaded model ready for PEFT.

== Usage Examples ==

=== Standard Loading ===
```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",
    torch_dtype=torch.float16,
)
```

=== Load for QLoRA ===
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

== Auto Class Variants ==

| Class | Use Case |
|-------|----------|
| `AutoModelForCausalLM` | Text generation (GPT, Llama) |
| `AutoModelForSeq2SeqLM` | Encoder-decoder (T5, BART) |
| `AutoModelForSequenceClassification` | Classification tasks |
| `AutoModelForTokenClassification` | NER, tagging |

== Related Functions ==

* [[huggingface_peft_get_peft_model]] - Apply PEFT after loading
* [[huggingface_peft_BitsAndBytesConfig]] - Quantization config

== Related Pages ==
* [[implemented_by::Principle:huggingface_peft_Model_Loading]]
* [[requires_env::Environment:huggingface_peft_CUDA_Training]]

[[Category:Implementation]]
[[Category:Wrapper_Doc]]
[[Category:External]]
