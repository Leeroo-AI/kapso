{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Attention Is All You Need|https://arxiv.org/abs/1706.03762]]
* [[source::Doc|HuggingFace Training Arguments|https://huggingface.co/docs/transformers/main_classes/trainer]]
* [[source::Blog|Learning Rate Scheduling|https://huggingface.co/docs/transformers/training]]
|-
! Domains
| [[domain::Optimization]], [[domain::Fine_Tuning]], [[domain::LLMs]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Strategy for selecting warmup steps to gradually ramp up learning rate at training start, preventing early instability.

=== Description ===
Warmup gradually increases the learning rate from 0 to the target value over a number of steps. This prevents large gradient updates early in training when the model hasn't yet found a stable region of the loss landscape. Particularly important for fine-tuning large models where initial gradients can be noisy.

=== Usage ===
Use this heuristic when configuring `warmup_steps` or `warmup_ratio` in training arguments. Essential for stable fine-tuning, especially with higher learning rates.

== The Insight (Rule of Thumb) ==
* **Action:** Set `warmup_steps` based on training duration and dataset size.
* **Values:**

{| class="wikitable"
! Training Duration !! Recommended Warmup !! Alternative
|-
|| Short (<100 steps) || 5-10 steps || warmup_ratio = 0.1
|-
|| Medium (100-1000 steps) || 10-50 steps || warmup_ratio = 0.05
|-
|| Long (>1000 steps) || 50-100 steps || warmup_ratio = 0.03
|}

* **Default Recommendation:** `warmup_steps = 10` for most Unsloth workflows
* **Quick Rule:** ~5% of total training steps, minimum 5 steps

<syntaxhighlight lang="python">
from trl import SFTConfig

args = SFTConfig(
    warmup_steps = 10,             # Fixed number of steps
    # OR
    warmup_ratio = 0.05,           # 5% of total steps
    lr_scheduler_type = "linear",  # or "cosine"
    # ... other args
)
</syntaxhighlight>

* **Trade-off:**
  * More warmup: Safer start, slightly slower to reach peak performance
  * Less warmup: Faster to peak LR, risk of instability

== Reasoning ==
At training start:
1. LoRA adapter weights are randomly initialized
2. Gradients can be large and noisy
3. Large LR × large gradients = unstable updates

Warmup allows the model to find a stable optimization trajectory before applying the full learning rate. For short fine-tuning runs (common with Unsloth), 5-10 steps is usually sufficient.

== Related Pages ==
=== Used By ===
* [[uses_heuristic::Workflow:QLoRA_Finetuning]]
* [[uses_heuristic::Implementation:TRL_SFTConfig]]
* [[uses_heuristic::Implementation:HF_TrainingArguments]]

