# Implementation: huggingface_peft_BitsAndBytesConfig

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Quantization|https://huggingface.co/docs/transformers/quantization]]
* [[source::Repo|bitsandbytes|https://github.com/TimDettmers/bitsandbytes]]
|-
! Domains
| [[domain::Quantization]], [[domain::QLoRA]], [[domain::External]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

| Property | Value |
|----------|-------|
| **Type** | Wrapper Doc |
| **Source** | External (transformers library) |
| **Principle** | [[implemented_by::Implementation:huggingface_peft_BitsAndBytesConfig]] |
| **Environment** | [[requires_env::Environment:huggingface_peft_Quantized_Training]] |

== Description ==

`BitsAndBytesConfig` configures 4-bit or 8-bit quantization for loading models with reduced memory. This is the foundation for QLoRA training, enabling fine-tuning of large models on consumer hardware.

== API Signature ==

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    bnb_4bit_quant_type: str = "fp4",
    bnb_4bit_compute_dtype: torch.dtype = torch.float32,
    bnb_4bit_use_double_quant: bool = False,
    bnb_4bit_quant_storage: torch.dtype = None,
)
```

== Parameters ==

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `load_in_4bit` | bool | False | Enable 4-bit quantization |
| `load_in_8bit` | bool | False | Enable 8-bit quantization |
| `bnb_4bit_quant_type` | str | "fp4" | Quantization type: "fp4" or "nf4" |
| `bnb_4bit_compute_dtype` | dtype | float32 | Computation dtype |
| `bnb_4bit_use_double_quant` | bool | False | Nested quantization |

== Quantization Types ==

| Type | Description |
|------|-------------|
| `fp4` | 4-bit floating point |
| `nf4` | NormalFloat 4-bit (better for normal distributions) |

== Usage Examples ==

=== Standard QLoRA Config ===
```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,  # Extra memory savings
)
```

=== 8-bit Quantization ===
```python
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
```

== Memory Savings ==

| Model Size | FP16 | 8-bit | 4-bit |
|------------|------|-------|-------|
| 7B | ~14GB | ~7GB | ~4GB |
| 13B | ~26GB | ~13GB | ~7GB |
| 70B | ~140GB | ~70GB | ~35GB |

== Related Functions ==

* [[huggingface_peft_AutoModel_from_pretrained]] - Use config when loading
* [[huggingface_peft_prepare_model_for_kbit_training]] - Prepare for training

== Related Pages ==
* [[implemented_by::Principle:huggingface_peft_Quantization_Config]]
* [[requires_env::Environment:huggingface_peft_Quantized_Training]]
* [[uses_heuristic::Heuristic:huggingface_peft_4bit_Defensive_Clone]]

[[Category:Implementation]]
[[Category:Wrapper_Doc]]
[[Category:Quantization]]
