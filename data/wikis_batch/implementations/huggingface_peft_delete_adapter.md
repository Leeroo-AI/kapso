# Implementation: huggingface_peft_delete_adapter

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|Multi-Adapter|https://huggingface.co/docs/peft/developer_guides/multi_adapter]]
|-
! Domains
| [[domain::Multi_Adapter]], [[domain::Memory_Management]], [[domain::Parameter_Efficient]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

| Property | Value |
|----------|-------|
| **Type** | API Doc |
| **Source** | `src/peft/peft_model.py` |
| **Principle** | [[implemented_by::Implementation:huggingface_peft_delete_adapter]] |
| **Environment** | [[requires_env::Environment:huggingface_peft_CUDA_Training]] |

== Description ==

`delete_adapter()` removes an adapter from the model, freeing its memory. This is useful for managing memory when cycling through many adapters or when an adapter is no longer needed.

== API Signature ==

```python
model.delete_adapter(adapter_name: str)
```

== Parameters ==

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adapter_name` | str | required | Name of adapter to delete |

== Returns ==

None - Modifies model in place, removes adapter.

== Usage Examples ==

=== Delete Single Adapter ===
```python
# Check loaded adapters
print(model.peft_config.keys())  # ['task_a', 'task_b', 'task_c']

# Delete one
model.delete_adapter("task_b")

print(model.peft_config.keys())  # ['task_a', 'task_c']
```

=== Cycle Through Adapters ===
```python
# Load and process each adapter, then delete
for adapter_path in adapter_paths:
    model.load_adapter(adapter_path, adapter_name="temp")
    model.set_adapter("temp")
    process(model)
    model.delete_adapter("temp")  # Free memory
```

== Key Behavior ==

1. **Removes adapter** - Deletes adapter structure and weights
2. **Frees memory** - GPU memory is released
3. **Cannot delete active** - Must switch away first if active

== Warnings ==

* Cannot delete the currently active adapter
* Cannot delete the last adapter (would leave model in invalid state)

== Related Functions ==

* [[huggingface_peft_load_adapter]] - Load adapters
* [[huggingface_peft_set_adapter]] - Switch before deleting active

== Related Pages ==
* [[implemented_by::Principle:huggingface_peft_Adapter_Lifecycle]]
* [[requires_env::Environment:huggingface_peft_CUDA_Training]]

[[Category:Implementation]]
[[Category:API_Doc]]
[[Category:Multi_Adapter]]
