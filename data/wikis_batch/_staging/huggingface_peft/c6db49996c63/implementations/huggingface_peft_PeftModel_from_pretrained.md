# Implementation: huggingface_peft_PeftModel_from_pretrained

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|PeftModel|https://huggingface.co/docs/peft/package_reference/peft_model]]
|-
! Domains
| [[domain::Model_Loading]], [[domain::Inference]], [[domain::Parameter_Efficient]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

| Property | Value |
|----------|-------|
| **Type** | API Doc |
| **Source** | `src/peft/peft_model.py:L389-700` |
| **Principle** | [[implemented_by::Implementation:huggingface_peft_PeftModel_from_pretrained]] |
| **Environment** | [[requires_env::Environment:huggingface_peft_CUDA_Training]] |

== Description ==

`PeftModel.from_pretrained()` loads a trained adapter and attaches it to a base model. Supports loading from local paths or HuggingFace Hub. The adapter configuration is automatically read from the saved checkpoint.

== API Signature ==

```python
from peft import PeftModel

peft_model = PeftModel.from_pretrained(
    model: PreTrainedModel,
    model_id: str,
    adapter_name: str = "default",
    is_trainable: bool = False,
    config: Optional[PeftConfig] = None,
    revision: Optional[str] = None,
    **kwargs
)
```

== Parameters ==

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | PreTrainedModel | required | Base model to attach adapter to |
| `model_id` | str | required | Path or Hub ID of adapter |
| `adapter_name` | str | "default" | Name for loaded adapter |
| `is_trainable` | bool | False | Whether to continue training |
| `config` | PeftConfig | None | Override saved config |
| `revision` | str | None | Git revision for Hub models |

== Returns ==

`PeftModel` - The model with loaded adapter attached.

== Usage Examples ==

=== Load from Hub ===
```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = PeftModel.from_pretrained(base_model, "username/my-lora-adapter")
```

=== Load from Local Path ===
```python
model = PeftModel.from_pretrained(base_model, "./my_adapter")
```

=== Continue Training ===
```python
model = PeftModel.from_pretrained(
    base_model,
    "username/my-adapter",
    is_trainable=True
)
```

== Key Behavior ==

1. **Loads config** - Reads `adapter_config.json` from checkpoint
2. **Injects adapter** - Creates adapter structure matching config
3. **Loads weights** - Loads `adapter_model.safetensors` into adapter
4. **Sets mode** - Inference mode by default (`is_trainable=False`)

== Related Functions ==

* [[huggingface_peft_save_pretrained]] - Save adapters for later loading
* [[huggingface_peft_load_adapter]] - Load additional adapters

== Related Pages ==
* [[implemented_by::Principle:huggingface_peft_Adapter_Loading]]
* [[requires_env::Environment:huggingface_peft_CUDA_Training]]

[[Category:Implementation]]
[[Category:API_Doc]]
[[Category:Model_Loading]]
