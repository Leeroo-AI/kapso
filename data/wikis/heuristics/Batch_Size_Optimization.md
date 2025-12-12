{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Unsloth Documentation|https://docs.unsloth.ai/]]
* [[source::Paper|Large Batch Training|https://arxiv.org/abs/1706.02677]]
* [[source::Blog|Effective Batch Size Guide|https://huggingface.co/docs/transformers/perf_train_gpu_one]]
|-
! Domains
| [[domain::Optimization]], [[domain::Memory_Management]], [[domain::LLMs]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Strategy for selecting batch size and gradient accumulation steps to maximize throughput while fitting in GPU memory.

=== Description ===
Batch size directly impacts training speed, memory usage, and model convergence. Unsloth's optimizations allow 2x larger batch sizes than standard implementations, enabling more efficient training. The effective batch size is `per_device_batch_size × gradient_accumulation_steps × num_gpus`.

=== Usage ===
Use this heuristic when configuring `per_device_train_batch_size` and `gradient_accumulation_steps` in `SFTConfig`. Critical for balancing speed and memory constraints.

== The Insight (Rule of Thumb) ==
* **Action:** Maximize `per_device_train_batch_size` until near OOM, then use gradient accumulation.
* **Values:**

{| class="wikitable"
! GPU VRAM !! Batch Size (7B model) !! With Unsloth
|-
|| 8GB || OOM || 1-2
|-
|| 12GB || 1 || 2-4
|-
|| 16GB || 1-2 || 4-8
|-
|| 24GB || 2-4 || 8-16
|-
|| 40GB+ || 8-16 || 16-32
|}

* **Target Effective Batch Size:** 32-128 for most tasks
* **Formula:** `effective_batch = per_device_batch × grad_accum × num_gpus`

<syntaxhighlight lang="python">
from trl import SFTConfig

# Example: 16GB GPU, targeting effective batch of 32
args = SFTConfig(
    per_device_train_batch_size = 4,     # Max that fits in memory
    gradient_accumulation_steps = 8,     # 4 * 8 = 32 effective
    # ... other args
)
</syntaxhighlight>

* **Trade-off:**
  * Larger batch: Faster training, more stable gradients, more VRAM
  * Smaller batch + accumulation: Less VRAM, same effective batch, slightly slower

== Reasoning ==
Larger effective batch sizes provide more stable gradient estimates, leading to smoother training curves. However, very large batches may require learning rate scaling. Unsloth's 70% VRAM reduction allows fitting larger batches than standard HuggingFace training, effectively doubling throughput.

Start with the largest batch that fits, then use gradient accumulation to reach the target effective batch size of 32-64 for most instruction tuning tasks.

== Related Pages ==
=== Used By ===
* [[uses_heuristic::Workflow:QLoRA_Finetuning]]
* [[uses_heuristic::Implementation:TRL_SFTConfig]]
* [[uses_heuristic::Implementation:HF_TrainingArguments]]
* [[uses_heuristic::Principle:Gradient_Checkpointing]]

