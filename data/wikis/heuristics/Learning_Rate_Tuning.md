{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|LoRA Paper|https://arxiv.org/abs/2106.09685]]
* [[source::Doc|Unsloth Documentation|https://docs.unsloth.ai/]]
* [[source::Blog|HuggingFace Training Tips|https://huggingface.co/docs/transformers/training]]
|-
! Domains
| [[domain::Optimization]], [[domain::Fine_Tuning]], [[domain::LLMs]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Optimal learning rate selection for LoRA/QLoRA fine-tuning, typically 10-100x higher than full fine-tuning rates.

=== Description ===
Learning rate is one of the most critical hyperparameters for successful fine-tuning. LoRA adapters require higher learning rates than full fine-tuning because they only update a small subset of parameters. Too low results in slow convergence; too high causes instability and loss spikes.

=== Usage ===
Use this heuristic when setting the `learning_rate` in `TrainingArguments` or `SFTConfig`. Essential for all fine-tuning workflows with Unsloth.

== The Insight (Rule of Thumb) ==
* **Action:** Set learning rate based on task type and model size.
* **Values:**

{| class="wikitable"
! Task Type !! Recommended LR !! Notes
|-
|| Instruction Fine-tuning (SFT) || 2e-4 to 5e-4 || Standard for most tasks
|-
|| Domain Adaptation || 1e-4 to 2e-4 || More conservative
|-
|| RLHF/DPO/GRPO || 5e-5 to 2e-4 || Lower for stability
|-
|| Small Models (<3B) || 3e-4 to 1e-3 || Can use higher LR
|-
|| Large Models (>13B) || 1e-4 to 2e-4 || More conservative
|}

* **Default Recommendation:** Start with `2e-4` for most QLoRA fine-tuning tasks.
* **Trade-off:** Higher LR = faster convergence but risk of instability; Lower LR = slower but more stable.

<syntaxhighlight lang="python">
from trl import SFTConfig

args = SFTConfig(
    learning_rate = 2e-4,           # Recommended starting point
    lr_scheduler_type = "linear",   # or "cosine" for longer training
    # ... other args
)
</syntaxhighlight>

== Reasoning ==
LoRA fine-tuning introduces new low-rank matrices that start from random initialization (or zeros for B matrices). These new parameters need to quickly adapt to the task, requiring higher learning rates. The base model weights remain frozen, so there's no risk of catastrophic forgetting from high LR.

Empirically, 2e-4 works well across most Llama, Mistral, and Qwen models when using 4-bit quantization with Unsloth.

== Related Pages ==
=== Used By ===
* [[uses_heuristic::Workflow:QLoRA_Finetuning]]
* [[uses_heuristic::Workflow:GRPO_Reinforcement_Learning]]
* [[uses_heuristic::Workflow:DPO_Alignment]]
* [[uses_heuristic::Implementation:TRL_SFTConfig]]
* [[uses_heuristic::Implementation:HF_TrainingArguments]]

