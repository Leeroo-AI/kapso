# Implementation: huggingface_peft_set_adapter

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|Multi-Adapter|https://huggingface.co/docs/peft/developer_guides/multi_adapter]]
|-
! Domains
| [[domain::Multi_Adapter]], [[domain::Inference]], [[domain::Parameter_Efficient]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

| Property | Value |
|----------|-------|
| **Type** | API Doc |
| **Source** | `src/peft/peft_model.py` |
| **Principle** | [[implemented_by::Implementation:huggingface_peft_set_adapter]] |
| **Environment** | [[requires_env::Environment:huggingface_peft_CUDA_Training]] |

== Description ==

`set_adapter()` activates one or more loaded adapters for the next forward pass. This enables dynamic switching between adapters without reloading weights, ideal for multi-task serving scenarios.

== API Signature ==

```python
model.set_adapter(adapter_name: Union[str, List[str]])
```

== Parameters ==

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adapter_name` | str / List[str] | required | Adapter(s) to activate |

== Returns ==

None - Modifies model state in place.

== Usage Examples ==

=== Switch Between Adapters ===
```python
# Assume adapters "task_a" and "task_b" are loaded
model.set_adapter("task_a")
output_a = model.generate(input_ids)

model.set_adapter("task_b")
output_b = model.generate(input_ids)
```

=== Activate Multiple Adapters ===
```python
# Use multiple adapters simultaneously (weights are summed)
model.set_adapter(["task_a", "task_b"])
```

=== Disable All Adapters ===
```python
with model.disable_adapter():
    # Uses base model only
    base_output = model.generate(input_ids)
```

== Key Behavior ==

1. **Activates adapter** - Sets specified adapter as active
2. **Zero overhead** - No weight loading, just pointer change
3. **Multiple adapters** - Can activate multiple (outputs summed)

== Related Functions ==

* [[huggingface_peft_load_adapter]] - Load adapters before switching
* [[huggingface_peft_delete_adapter]] - Remove adapters
* [[huggingface_peft_add_weighted_adapter]] - Combine adapters with weights

== Related Pages ==
* [[implemented_by::Principle:huggingface_peft_Adapter_Switching]]
* [[requires_env::Environment:huggingface_peft_CUDA_Training]]

[[Category:Implementation]]
[[Category:API_Doc]]
[[Category:Multi_Adapter]]
