# Implementation: huggingface_peft_get_peft_model

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|PEFT API|https://huggingface.co/docs/peft/package_reference/peft_model]]
|-
! Domains
| [[domain::Model_Creation]], [[domain::PEFT]], [[domain::Parameter_Efficient]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

| Property | Value |
|----------|-------|
| **Type** | API Doc |
| **Source** | `src/peft/mapping.py` |
| **Principle** | [[implemented_by::Implementation:huggingface_peft_get_peft_model]] |
| **Environment** | [[requires_env::Environment:huggingface_peft_CUDA_Training]] |

== Description ==

`get_peft_model()` is the primary factory function for creating PEFT models. It wraps a pretrained model with parameter-efficient adapters based on the provided configuration, freezing the base model weights and making only the adapter parameters trainable.

== API Signature ==

```python
from peft import get_peft_model

peft_model = get_peft_model(
    model: PreTrainedModel,
    peft_config: PeftConfig,
    adapter_name: str = "default",
    mixed: bool = False,
    autocast_adapter_dtype: bool = True,
    revision: Optional[str] = None,
)
```

== Parameters ==

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | PreTrainedModel | required | The base model to wrap |
| `peft_config` | PeftConfig | required | Configuration (LoraConfig, etc.) |
| `adapter_name` | str | "default" | Name for the adapter |
| `mixed` | bool | False | Whether to use mixed precision |
| `autocast_adapter_dtype` | bool | True | Auto-cast adapter to model dtype |

== Returns ==

`PeftModel` - The wrapped model with adapters injected and base weights frozen.

== Usage Examples ==

=== Basic Usage ===
```python
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear")

peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622
```

=== Named Adapter ===
```python
peft_model = get_peft_model(model, config, adapter_name="task_a")
```

== Key Behavior ==

1. **Freezes base model** - Sets `requires_grad=False` on all base parameters
2. **Injects adapters** - Adds LoRA matrices to target modules
3. **Routes forward pass** - Modified forward: `output = base_output + adapter_output`

== Related Functions ==

* [[huggingface_peft_LoraConfig]] - Configuration for LoRA adapters
* [[huggingface_peft_PeftModel_from_pretrained]] - Load existing adapters

== Related Pages ==
* [[implemented_by::Principle:huggingface_peft_PEFT_Application]]
* [[requires_env::Environment:huggingface_peft_CUDA_Training]]

[[Category:Implementation]]
[[Category:API_Doc]]
[[Category:PEFT]]
