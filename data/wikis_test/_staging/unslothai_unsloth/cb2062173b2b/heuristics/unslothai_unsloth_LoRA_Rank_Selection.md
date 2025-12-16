# Heuristic: unslothai_unsloth_LoRA_Rank_Selection

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Paper|LoRA Paper|https://arxiv.org/abs/2106.09685]]
* [[source::Discussion|Unsloth Discord|https://discord.gg/unsloth]]
|-
! Domains
| [[domain::LoRA]], [[domain::Fine_Tuning]], [[domain::Memory_Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-16 12:00 GMT]]
|}

## Overview

Guidance for selecting LoRA rank (`r`) parameter based on task complexity, balancing trainable parameters against learning capacity.

### Description

The LoRA rank determines the dimensionality of the low-rank decomposition matrices (A and B) that approximate weight updates. Higher ranks increase expressiveness but also increase:
- Memory usage (more trainable parameters)
- Training time
- Risk of overfitting on small datasets

Unsloth's default `r=16` works well for most tasks, with `max_lora_rank=64` as the upper bound for vLLM fast inference.

### Usage

Use this heuristic when:
- Configuring `FastLanguageModel.get_peft_model(r=...)`
- Balancing between model quality and resource constraints
- Debugging poor fine-tuning results (underfitting vs overfitting)

## The Insight (Rule of Thumb)

* **Action:** Set `r` parameter in `get_peft_model()` based on task complexity
* **Default Value:** `r=16` (Unsloth default)
* **Recommended Ranges:**
  - **Simple tasks** (classification, basic instruction): `r=8-16`
  - **Standard fine-tuning** (chat, Q&A): `r=16-32`
  - **Complex reasoning** (math, code, long-form): `r=32-64`
  - **Full capability transfer**: `r=64-128` (may need more VRAM)
* **Trade-off:** Higher rank = better learning capacity but 2x memory per doubling of r
* **Constraint:** `max_lora_rank=64` when using `fast_inference=True` (vLLM)

### Target Modules

Default Unsloth target modules cover all major attention and MLP projections:
```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",     # MLP
]
```

## Reasoning

From the LoRA paper: Low-rank updates capture most of the adaptation signal. Empirically:
- r=8 captures ~95% of adaptation for simple tasks
- r=64 needed for complex multi-step reasoning
- Beyond r=128, diminishing returns unless very large dataset

The trainable parameter count scales as: `2 * r * d * num_layers * num_modules` where `d` is the hidden dimension. A 7B model with r=16 trains ~0.2% of parameters; r=64 trains ~0.8%.

## Code Evidence

From `unsloth/models/llama.py:2578-2604`:
```python
@staticmethod
def get_peft_model(
    model,
    r = 16,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = 16,
    # ...
):
    if type(r) is not int:
        raise TypeError(f"Unsloth: Rank of {str(r)} must be an integer.")
    if r <= 0:
        raise TypeError(f"Unsloth: Rank of {str(r)} must be larger than 0.")
```

vLLM constraint from `unsloth/models/loader.py:145`:
```python
max_lora_rank = 64,  # Maximum rank for vLLM fast inference
```

## Related Pages

* [[uses_heuristic::Implementation:unslothai_unsloth_get_peft_model]]
* [[uses_heuristic::Workflow:unslothai_unsloth_QLoRA_Finetuning]]
* [[uses_heuristic::Principle:unslothai_unsloth_LoRA_Configuration]]
