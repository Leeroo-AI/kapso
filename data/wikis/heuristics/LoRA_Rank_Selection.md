{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|LoRA Paper|https://arxiv.org/abs/2106.09685]]
* [[source::Paper|QLoRA Paper|https://arxiv.org/abs/2305.14314]]
* [[source::Doc|Unsloth Documentation|https://docs.unsloth.ai/]]
|-
! Domains
| [[domain::Fine_Tuning]], [[domain::Optimization]], [[domain::LLMs]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Guidelines for selecting LoRA rank (r) to balance model capacity, training cost, and fine-tuning quality.

=== Description ===
LoRA rank determines the dimensionality of the low-rank adaptation matrices. Higher rank means more trainable parameters and greater capacity to learn task-specific patterns, but also more memory and compute. The optimal rank depends on task complexity and the gap between pre-training and target domain.

=== Usage ===
Use this heuristic when setting the `r` parameter in `FastLanguageModel.get_peft_model()`. Affects model quality, training speed, and adapter size.

== The Insight (Rule of Thumb) ==
* **Action:** Set `r` based on task complexity and available resources.
* **Values:**

{| class="wikitable"
! Rank (r) !! Use Case !! Trainable Params (7B) !! Quality
|-
|| 8 || Simple tasks, quick experiments || ~20M || Good
|-
|| 16 || Standard instruction tuning || ~40M || Better (recommended)
|-
|| 32 || Complex tasks, domain adaptation || ~80M || Best
|-
|| 64+ || Near full fine-tuning capacity || ~160M+ || Diminishing returns
|}

* **Default Recommendation:** `r = 16` for most tasks
* **Alpha Rule:** Set `lora_alpha = r` or `lora_alpha = 2 * r`

<syntaxhighlight lang="python">
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,              # Rank - start here
    lora_alpha = 16,     # Usually equal to r
    lora_dropout = 0,    # 0 is optimized for Unsloth
    bias = "none",
    # ... other args
)
</syntaxhighlight>

* **Trade-off:**
  * Higher rank: More capacity, more VRAM, larger adapter files
  * Lower rank: Faster training, smaller adapters, may underfit complex tasks

== Reasoning ==
The LoRA paper showed that language models can be effectively adapted with surprisingly low ranks. Rank 16-32 captures most task-specific information for instruction tuning. Higher ranks show diminishing returns because the "intrinsic rank" of the update is often low.

For Unsloth:
* Use `r = 8` for quick experiments and simple classification
* Use `r = 16` for standard instruction tuning (default)
* Use `r = 32` for complex reasoning tasks or significant domain shift

== Related Pages ==
=== Used By ===
* [[uses_heuristic::Workflow:QLoRA_Finetuning]]
* [[uses_heuristic::Implementation:Unsloth_get_peft_model]]
* [[uses_heuristic::Principle:Low_Rank_Adaptation]]

