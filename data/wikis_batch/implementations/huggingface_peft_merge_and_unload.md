# Implementation: huggingface_peft_merge_and_unload

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|Merging|https://huggingface.co/docs/peft/developer_guides/model_merging]]
|-
! Domains
| [[domain::Model_Merging]], [[domain::Deployment]], [[domain::Parameter_Efficient]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

| Property | Value |
|----------|-------|
| **Type** | API Doc |
| **Source** | `src/peft/peft_model.py` |
| **Principle** | [[implemented_by::Implementation:huggingface_peft_merge_and_unload]] |
| **Environment** | [[requires_env::Environment:huggingface_peft_CUDA_Training]] |

== Description ==

`merge_and_unload()` permanently merges adapter weights into the base model and removes the PEFT wrapper. The result is a standard PyTorch model with the adapter's learned behavior baked in, useful for deployment without PEFT dependencies.

== API Signature ==

```python
merged_model = model.merge_and_unload(
    progressbar: bool = False,
    safe_merge: bool = False,
    adapter_names: Optional[List[str]] = None,
)
```

== Parameters ==

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `progressbar` | bool | False | Show merge progress |
| `safe_merge` | bool | False | Check for NaN after merge |
| `adapter_names` | List[str] | None | Specific adapters to merge (None=active) |

== Returns ==

`PreTrainedModel` - Standard model with merged weights (no PEFT wrapper).

== Usage Examples ==

=== Basic Merge ===
```python
from peft import PeftModel

# Load model with adapter
model = PeftModel.from_pretrained(base_model, "my-adapter")

# Merge and get standard model
merged_model = model.merge_and_unload()

# Save as standard HuggingFace model
merged_model.save_pretrained("./merged_model")
```

=== Safe Merge (Detect Broken Adapters) ===
```python
merged_model = model.merge_and_unload(safe_merge=True)
# Raises ValueError if NaN detected in merged weights
```

== Key Behavior ==

1. **Merges weights** - Computes `W_merged = W_base + B @ A * scale`
2. **Removes PEFT layers** - Unwraps adapter structure
3. **Returns base model** - Standard PreTrainedModel without PEFT

== Warnings ==

* **Quantized models** - Merging into quantized base introduces rounding errors
* **Irreversible** - Cannot separate adapter after merge

== Related Functions ==

* [[huggingface_peft_PeftModel_from_pretrained]] - Load before merging
* [[huggingface_peft_save_pretrained]] - Alternative: save adapter separately

== Related Pages ==
* [[implemented_by::Principle:huggingface_peft_Adapter_Merging]]
* [[requires_env::Environment:huggingface_peft_CUDA_Training]]
* [[uses_heuristic::Heuristic:huggingface_peft_Quantized_Merge_Rounding]]
* [[uses_heuristic::Heuristic:huggingface_peft_Safe_Merge_NaN_Check]]

[[Category:Implementation]]
[[Category:API_Doc]]
[[Category:Model_Merging]]
