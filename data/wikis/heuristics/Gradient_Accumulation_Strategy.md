{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|HuggingFace Trainer|https://huggingface.co/docs/transformers/main_classes/trainer]]
* [[source::Doc|Unsloth Documentation|https://docs.unsloth.ai/]]
* [[source::Blog|Gradient Accumulation Guide|https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation]]
|-
! Domains
| [[domain::Optimization]], [[domain::Memory_Management]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Technique to simulate larger batch sizes by accumulating gradients over multiple forward passes before updating weights.

=== Description ===
Gradient accumulation allows training with larger effective batch sizes than GPU memory permits. Instead of updating weights after each batch, gradients are accumulated over `N` steps before performing a single update. This achieves the same effect as training with `N × batch_size` samples per update.

=== Usage ===
Use this heuristic when you need **larger effective batch sizes** than memory allows, or when you want to match batch sizes used in paper reproductions. Standard practice for all Unsloth fine-tuning workflows.

== The Insight (Rule of Thumb) ==
* **Action:** Set `gradient_accumulation_steps` to achieve target effective batch size.
* **Formula:** `effective_batch = per_device_batch × gradient_accumulation_steps`
* **Values:**

{| class="wikitable"
! Target Effective Batch !! Per-Device Batch !! Accumulation Steps
|-
|| 32 || 2 || 16
|-
|| 32 || 4 || 8
|-
|| 32 || 8 || 4
|-
|| 64 || 4 || 16
|-
|| 64 || 8 || 8
|}

<syntaxhighlight lang="python">
from trl import SFTConfig

args = SFTConfig(
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,  # Effective batch = 8
    # ... other args
)
</syntaxhighlight>

* **Trade-off:**
  * More accumulation steps: Same memory as small batch, but slower per-step
  * Fewer accumulation steps: Need more VRAM, but faster wall-clock time

== Reasoning ==
Mathematically, accumulating gradients over N steps and averaging them produces identical gradients to computing them on N × batch_size samples at once. This allows:

1. Training with large effective batches on limited hardware
2. Matching batch sizes from papers for reproducibility
3. More stable training through better gradient estimates

With Unsloth's 70% VRAM reduction, you often need less accumulation than with standard HuggingFace, leading to faster training.

== Related Pages ==
=== Used By ===
* [[uses_heuristic::Workflow:QLoRA_Finetuning]]
* [[uses_heuristic::Implementation:TRL_SFTTrainer]]
* [[uses_heuristic::Implementation:TRL_SFTConfig]]
* [[uses_heuristic::Implementation:HF_TrainingArguments]]

