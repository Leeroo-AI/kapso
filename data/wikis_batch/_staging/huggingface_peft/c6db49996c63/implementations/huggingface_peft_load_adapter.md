# Implementation: huggingface_peft_load_adapter

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|Multi-Adapter|https://huggingface.co/docs/peft/developer_guides/multi_adapter]]
|-
! Domains
| [[domain::Multi_Adapter]], [[domain::Model_Loading]], [[domain::Parameter_Efficient]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

| Property | Value |
|----------|-------|
| **Type** | API Doc |
| **Source** | `src/peft/peft_model.py` |
| **Principle** | [[implemented_by::Implementation:huggingface_peft_load_adapter]] |
| **Environment** | [[requires_env::Environment:huggingface_peft_CUDA_Training]] |

== Description ==

`load_adapter()` loads an additional adapter into an existing PeftModel. This enables multi-adapter scenarios where different adapters can be loaded and switched between for different tasks or domains.

== API Signature ==

```python
model.load_adapter(
    model_id: str,
    adapter_name: str,
    is_trainable: bool = False,
    torch_device: Optional[str] = None,
    **kwargs
)
```

== Parameters ==

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | str | required | Path or Hub ID of adapter to load |
| `adapter_name` | str | required | Unique name for this adapter |
| `is_trainable` | bool | False | Whether adapter should be trainable |
| `torch_device` | str | None | Device to load adapter to |

== Returns ==

None - Modifies model in place.

== Usage Examples ==

=== Load Multiple Adapters ===
```python
from peft import PeftModel

# Load first adapter
model = PeftModel.from_pretrained(base_model, "adapter_a", adapter_name="task_a")

# Load additional adapters
model.load_adapter("adapter_b", adapter_name="task_b")
model.load_adapter("adapter_c", adapter_name="task_c")

# Check loaded adapters
print(model.peft_config.keys())  # ['task_a', 'task_b', 'task_c']
```

=== Load from Hub ===
```python
model.load_adapter("username/my-domain-adapter", adapter_name="domain")
```

== Key Behavior ==

1. **Loads config** - Reads adapter configuration from checkpoint
2. **Creates adapter** - Adds new adapter structure to model
3. **Loads weights** - Populates adapter with saved weights
4. **Does NOT activate** - Use `set_adapter()` to activate

== Related Functions ==

* [[huggingface_peft_set_adapter]] - Activate a loaded adapter
* [[huggingface_peft_delete_adapter]] - Remove a loaded adapter
* [[huggingface_peft_PeftModel_from_pretrained]] - Initial adapter loading

== Related Pages ==
* [[implemented_by::Principle:huggingface_peft_Adapter_Addition]]
* [[requires_env::Environment:huggingface_peft_CUDA_Training]]

[[Category:Implementation]]
[[Category:API_Doc]]
[[Category:Multi_Adapter]]
