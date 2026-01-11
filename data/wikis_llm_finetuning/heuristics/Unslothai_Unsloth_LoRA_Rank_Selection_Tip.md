# Heuristic: LoRA_Rank_Selection_Tip

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|loader.py|https://github.com/unslothai/unsloth/blob/main/unsloth/models/loader.py]]
|-
! Domains
| [[domain::Optimization]], [[domain::LoRA]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2026-01-09 12:00 GMT]]
|}

== Overview ==
Use LoRA rank between 8-128, with r=16 as a solid default for most fine-tuning tasks.

=== Description ===
The LoRA (Low-Rank Adaptation) rank parameter `r` controls the dimensionality of the low-rank matrices used for adaptation. Higher ranks increase model capacity but also increase memory usage and may lead to overfitting. Lower ranks are more parameter-efficient but may not capture complex adaptations.

The `lora_alpha` parameter should typically be set equal to `r` or `2*r` for stable training.

=== Usage ===
Use this heuristic when:
- **Choosing initial LoRA configuration:** Starting a new fine-tuning project
- **Debugging underfitting:** Model not learning task well → increase rank
- **Reducing memory usage:** Need to fit training in less VRAM → decrease rank
- **vLLM inference:** Must use rank ≤ 64 for vLLM LoRA support

== The Insight (Rule of Thumb) ==
* **Action:** Set `r` parameter in `get_peft_model()`
* **Value:**
  - **r=8**: Minimal, good for simple tasks or extremely limited VRAM
  - **r=16**: Solid default for most fine-tuning tasks
  - **r=32**: Better for complex tasks or chat models
  - **r=64**: Maximum for vLLM compatibility, good for complex reasoning tasks
  - **r=128+**: Only for non-vLLM training, may overfit
* **Trade-off:** Higher rank = more parameters = more VRAM = risk of overfitting
* **Alpha:** Set `lora_alpha = r` or `lora_alpha = 2*r`

== Reasoning ==
LoRA's effectiveness comes from the observation that the weight updates during fine-tuning have low intrinsic rank. The rank `r` determines how many dimensions of this low-rank subspace are captured:

1. **Too low (r < 8):** May not capture enough task-specific information
2. **Sweet spot (r = 16-64):** Balances capacity vs efficiency for most tasks
3. **Too high (r > 128):** Diminishing returns, increased overfitting risk

Research shows that r=16 captures most of the adaptation capacity needed for typical instruction-tuning tasks.

**vLLM Constraint:** When using `fast_inference=True` for GRPO or other RL workflows, vLLM's LoRA implementation has a maximum rank of 64. Exceeding this will cause runtime errors.

== Code Evidence ==

Default LoRA targets from `loader.py:795-803`:
<syntaxhighlight lang="python">
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
# ...
r = 16,
lora_alpha = 16,
</syntaxhighlight>

vLLM rank constraint from `rl.py:145-155`:
<syntaxhighlight lang="python">
max_lora_rank = lora_request.max_lora_rank if lora_request is not None else 64
# LoRA rank must not exceed vLLM's max_lora_rank
if r > max_lora_rank:
    raise ValueError(
        f"LoRA rank {r} exceeds vLLM max_lora_rank {max_lora_rank}"
    )
</syntaxhighlight>

== Related Pages ==
* [[used_by::Implementation:Unslothai_Unsloth_get_peft_model]]
* [[used_by::Implementation:Unslothai_Unsloth_get_peft_model_rl]]
