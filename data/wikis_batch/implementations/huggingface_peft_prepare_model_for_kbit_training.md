# Implementation: huggingface_peft_prepare_model_for_kbit_training

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|QLoRA|https://huggingface.co/docs/peft/developer_guides/quantization]]
|-
! Domains
| [[domain::QLoRA]], [[domain::Memory_Optimization]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

| Property | Value |
|----------|-------|
| **Type** | API Doc |
| **Source** | `src/peft/utils/other.py:L130-215` |
| **Principle** | [[implemented_by::Implementation:huggingface_peft_prepare_model_for_kbit_training]] |
| **Environment** | [[requires_env::Environment:huggingface_peft_Quantized_Training]] |

== Description ==

`prepare_model_for_kbit_training()` prepares a quantized model (4-bit or 8-bit) for training. It handles the necessary setup for gradient computation through quantized layers, including gradient checkpointing, layer norm casting, and input gradient enabling.

== API Signature ==

```python
from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(
    model: PreTrainedModel,
    use_gradient_checkpointing: bool = True,
    gradient_checkpointing_kwargs: Optional[dict] = None,
)
```

== Parameters ==

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | PreTrainedModel | required | Quantized model to prepare |
| `use_gradient_checkpointing` | bool | True | Enable gradient checkpointing |
| `gradient_checkpointing_kwargs` | dict | None | Args for checkpointing |

== Returns ==

`PreTrainedModel` - The prepared model ready for k-bit training.

== Usage Examples ==

=== QLoRA Setup ===
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig

# Load quantized model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
)

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# Apply LoRA
config = LoraConfig(r=16, target_modules="all-linear")
model = get_peft_model(model, config)
```

=== Disable Gradient Checkpointing ===
```python
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=False
)
```

== Key Behavior ==

1. **Casts layer norms to fp32** - Numerical stability
2. **Enables input gradients** - Required for backward pass
3. **Enables gradient checkpointing** - Memory optimization (default)
4. **Freezes base model** - Only adapters will train

== Related Functions ==

* [[huggingface_peft_BitsAndBytesConfig]] - Create quantization config
* [[huggingface_peft_get_peft_model]] - Apply LoRA after preparation

== Related Pages ==
* [[implemented_by::Principle:huggingface_peft_Memory_Optimization]]
* [[requires_env::Environment:huggingface_peft_Quantized_Training]]
* [[uses_heuristic::Heuristic:huggingface_peft_Gradient_Checkpointing]]

[[Category:Implementation]]
[[Category:API_Doc]]
[[Category:QLoRA]]
