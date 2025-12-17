# Implementation: huggingface_peft_add_weighted_adapter

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|Adapter Merging|https://huggingface.co/docs/peft/developer_guides/model_merging]]
|-
! Domains
| [[domain::Multi_Adapter]], [[domain::Model_Merging]], [[domain::Parameter_Efficient]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

| Property | Value |
|----------|-------|
| **Type** | API Doc |
| **Source** | `src/peft/tuners/lora/model.py` |
| **Principle** | [[implemented_by::Implementation:huggingface_peft_add_weighted_adapter]] |
| **Environment** | [[requires_env::Environment:huggingface_peft_CUDA_Training]] |

== Description ==

`add_weighted_adapter()` combines multiple LoRA adapters into a new adapter using weighted averaging. This enables creating task-interpolated adapters or ensembling multiple fine-tuned adapters.

== API Signature ==

```python
model.add_weighted_adapter(
    adapters: List[str],
    weights: List[float],
    adapter_name: str,
    combination_type: str = "linear",
    svd_rank: Optional[int] = None,
    svd_clamp: Optional[float] = None,
    svd_full_matrices: bool = True,
    svd_driver: Optional[str] = None,
    density: Optional[float] = None,
    majority_sign_method: str = "total",
)
```

== Parameters ==

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adapters` | List[str] | required | Adapter names to combine |
| `weights` | List[float] | required | Weight for each adapter |
| `adapter_name` | str | required | Name for new combined adapter |
| `combination_type` | str | "linear" | How to combine: "linear", "svd", "ties", "dare_ties" |
| `svd_rank` | int | None | Rank for SVD-based combination |
| `density` | float | None | For TIES/DARE: fraction of weights to keep |

== Combination Types ==

| Type | Description |
|------|-------------|
| `linear` | Simple weighted average: `sum(w_i * adapter_i)` |
| `svd` | SVD decomposition of combined weights |
| `ties` | TIES merging with sign consensus |
| `dare_ties` | DARE + TIES for sparse merging |

== Usage Examples ==

=== Linear Combination ===
```python
# Average two adapters equally
model.add_weighted_adapter(
    adapters=["task_a", "task_b"],
    weights=[0.5, 0.5],
    adapter_name="combined",
)
model.set_adapter("combined")
```

=== Weighted Combination ===
```python
# Favor task_a (70%) over task_b (30%)
model.add_weighted_adapter(
    adapters=["task_a", "task_b"],
    weights=[0.7, 0.3],
    adapter_name="mostly_a",
)
```

=== TIES Merging ===
```python
model.add_weighted_adapter(
    adapters=["adapter_1", "adapter_2", "adapter_3"],
    weights=[1.0, 1.0, 1.0],
    adapter_name="ties_merged",
    combination_type="ties",
    density=0.5,
)
```

== Key Behavior ==

1. **Creates new adapter** - Does not modify source adapters
2. **Requires same architecture** - Adapters must have same target modules
3. **Activatable** - Use `set_adapter()` to activate combined adapter

== Related Functions ==

* [[huggingface_peft_load_adapter]] - Load adapters to combine
* [[huggingface_peft_set_adapter]] - Activate combined adapter
* [[huggingface_peft_merge_and_unload]] - Merge into base model

== Related Pages ==
* [[implemented_by::Principle:huggingface_peft_Adapter_Combination]]
* [[requires_env::Environment:huggingface_peft_CUDA_Training]]

[[Category:Implementation]]
[[Category:API_Doc]]
[[Category:Multi_Adapter]]
