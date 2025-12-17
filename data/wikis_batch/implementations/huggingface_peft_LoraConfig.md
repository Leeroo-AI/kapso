# Implementation: huggingface_peft_LoraConfig

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|LoRA API|https://huggingface.co/docs/peft/package_reference/lora]]
|-
! Domains
| [[domain::Configuration]], [[domain::LoRA]], [[domain::Parameter_Efficient]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

| Property | Value |
|----------|-------|
| **Type** | API Doc |
| **Source** | `src/peft/tuners/lora/config.py:L322-880` |
| **Principle** | [[implemented_by::Implementation:huggingface_peft_LoraConfig]] |
| **Environment** | [[requires_env::Environment:huggingface_peft_CUDA_Training]] |

== Description ==

`LoraConfig` is the configuration class for Low-Rank Adaptation (LoRA) tuning. It defines all hyperparameters controlling how LoRA adapters are injected into transformer models including rank, alpha scaling, target modules, and advanced features like RSLoRA, DoRA, and weight decomposition.

== API Signature ==

```python
from peft import LoraConfig

config = LoraConfig(
    r: int = 8,
    lora_alpha: int = 8,
    target_modules: Optional[Union[List[str], str]] = None,
    lora_dropout: float = 0.0,
    fan_in_fan_out: bool = False,
    bias: str = "none",
    use_rslora: bool = False,
    modules_to_save: Optional[List[str]] = None,
    init_lora_weights: Union[bool, str] = True,
    layers_to_transform: Optional[Union[List[int], int]] = None,
    layers_pattern: Optional[Union[List[str], str]] = None,
    rank_pattern: Optional[dict] = None,
    alpha_pattern: Optional[dict] = None,
    use_dora: bool = False,
    task_type: Optional[str] = None,
)
```

== Parameters ==

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `r` | int | 8 | LoRA attention dimension (rank) |
| `lora_alpha` | int | 8 | Alpha parameter for LoRA scaling |
| `target_modules` | List[str] / str | None | Modules to apply LoRA. Use "all-linear" for all linear layers |
| `lora_dropout` | float | 0.0 | Dropout probability for LoRA layers |
| `bias` | str | "none" | Bias type: "none", "all", or "lora_only" |
| `use_rslora` | bool | False | Use Rank-Stabilized LoRA (scale by sqrt(r)) |
| `use_dora` | bool | False | Use Weight-Decomposed LRA (DoRA) |
| `task_type` | str | None | Task type (CAUSAL_LM, SEQ_2_SEQ_LM, etc.) |

== Usage Examples ==

=== Basic Configuration ===
```python
from peft import LoraConfig, TaskType

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)
```

=== All-Linear Targeting ===
```python
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    task_type=TaskType.CAUSAL_LM,
)
```

=== RSLoRA for Higher Ranks ===
```python
config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules="all-linear",
    use_rslora=True,  # Better stability at high ranks
    task_type=TaskType.CAUSAL_LM,
)
```

== Related Functions ==

* [[huggingface_peft_get_peft_model]] - Applies this config to a model
* [[huggingface_peft_save_pretrained]] - Saves config with adapter

== Related Pages ==
* [[implemented_by::Principle:huggingface_peft_LoRA_Configuration]]
* [[requires_env::Environment:huggingface_peft_CUDA_Training]]
* [[uses_heuristic::Heuristic:huggingface_peft_DoRA_Mixed_Batch_Limitation]]

[[Category:Implementation]]
[[Category:API_Doc]]
[[Category:LoRA]]
