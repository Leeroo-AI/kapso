# Implementation: huggingface_peft_save_pretrained

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|Save/Load|https://huggingface.co/docs/peft/developer_guides/checkpoint]]
|-
! Domains
| [[domain::Serialization]], [[domain::Checkpoint]], [[domain::Parameter_Efficient]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

| Property | Value |
|----------|-------|
| **Type** | API Doc |
| **Source** | `src/peft/peft_model.py:L190-387` |
| **Principle** | [[implemented_by::Implementation:huggingface_peft_save_pretrained]] |
| **Environment** | [[requires_env::Environment:huggingface_peft_CUDA_Training]] |

== Description ==

`save_pretrained()` saves adapter weights and configuration to disk. Only the adapter parameters are saved (typically a few MB), not the full model weights. Supports safetensors format for secure serialization.

== API Signature ==

```python
model.save_pretrained(
    save_directory: str,
    safe_serialization: bool = True,
    selected_adapters: Optional[List[str]] = None,
    save_embedding_layers: Union[str, bool] = "auto",
    is_main_process: bool = True,
    **kwargs
)
```

== Parameters ==

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_directory` | str | required | Directory to save adapter files |
| `safe_serialization` | bool | True | Use safetensors format |
| `selected_adapters` | List[str] | None | Specific adapters to save (None=all) |
| `save_embedding_layers` | str/bool | "auto" | Save embedding layers if modified |
| `is_main_process` | bool | True | Only save on main process (distributed) |

== Output Files ==

```
save_directory/
├── adapter_config.json       # LoRA configuration
├── adapter_model.safetensors # Adapter weights (or .bin if safe=False)
└── README.md                 # Model card (optional)
```

== Usage Examples ==

=== Basic Save ===
```python
# After training
model.save_pretrained("./my_adapter")
```

=== Push to Hub ===
```python
model.push_to_hub(
    "username/my-adapter",
    private=True,
    token="hf_..."
)
```

=== Save Specific Adapters ===
```python
model.save_pretrained(
    "./adapter_a",
    selected_adapters=["adapter_a"]
)
```

== Related Functions ==

* [[huggingface_peft_PeftModel_from_pretrained]] - Load saved adapters
* [[huggingface_peft_get_peft_model]] - Create model to save

== Related Pages ==
* [[implemented_by::Principle:huggingface_peft_Adapter_Saving]]
* [[requires_env::Environment:huggingface_peft_CUDA_Training]]

[[Category:Implementation]]
[[Category:API_Doc]]
[[Category:Serialization]]
