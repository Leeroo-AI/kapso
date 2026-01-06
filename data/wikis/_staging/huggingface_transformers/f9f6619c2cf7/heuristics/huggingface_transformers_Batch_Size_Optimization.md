# Heuristic: huggingface_transformers_Batch_Size_Optimization

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Performance Training|https://huggingface.co/docs/transformers/perf_train_gpu_one]]
|-
! Domains
| [[domain::Optimization]], [[domain::Training]], [[domain::Throughput]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==
Use gradient accumulation to simulate larger batch sizes when VRAM-constrained, achieving equivalent training dynamics without OOM errors.

=== Description ===
Effective batch size = `per_device_train_batch_size * gradient_accumulation_steps * num_devices`. When GPU memory is limited, reduce per-device batch size and increase gradient accumulation steps proportionally. This maintains the same effective batch size and training dynamics while fitting within VRAM constraints.

=== Usage ===
Use this when you need a specific effective batch size for training stability but cannot fit it in GPU memory. Common when fine-tuning large models or training with limited hardware.

== The Insight (Rule of Thumb) ==

* **Action:** Set `gradient_accumulation_steps = target_batch_size / (per_device_batch_size * num_gpus)`
* **Value:** Start with `per_device_train_batch_size=1` and increase `gradient_accumulation_steps` as needed
* **Trade-off:** Equivalent training dynamics; slightly lower GPU utilization due to smaller micro-batches
* **Best Practice:** Total batch size should be power of 2 (32, 64, 128, 256) for optimal GPU utilization

== Reasoning ==

Gradient accumulation allows training with effectively large batch sizes without holding all activations in memory simultaneously. The gradients are accumulated across multiple forward-backward passes before performing an optimizer step.

For LLM fine-tuning:
- Effective batch size of 32-64 works well for most tasks
- Per-device batch size of 1-4 is common with gradient checkpointing
- Gradient accumulation of 8-32 steps fills the gap

== Code Evidence ==

From `training_args.py:L238-246`:

<syntaxhighlight lang="python">
gradient_accumulation_steps (`int`, *optional*, defaults to 1):
    Number of updates steps to accumulate the gradients for, before
    performing a backward/update pass.

    <Tip warning={true}>

    When using gradient accumulation, one step is counted as one step with
    backward pass. Therefore, logging, evaluation, save will be conducted
    every `gradient_accumulation_steps * xxx_step` training examples.

    </Tip>
</syntaxhighlight>

From `examples/3D_parallel.py:L80`:

<syntaxhighlight lang="python">
global_batch_size = 8  # Desired global batch size
# ...
local_batch_size = global_batch_size // dp_mesh.size()
</syntaxhighlight>

== Example Usage ==

<syntaxhighlight lang="python">
from transformers import TrainingArguments

# Target: effective batch size of 64 on 2 GPUs
# Constraint: only 4 samples fit per GPU

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,      # What fits in VRAM
    gradient_accumulation_steps=8,       # 4 * 8 * 2 GPUs = 64
    # ... other args
)

# Calculation:
# effective_batch = per_device * accumulation * num_gpus
# 64 = 4 * 8 * 2
</syntaxhighlight>

== Common Patterns ==

{| class="wikitable"
|-
! Scenario !! per_device_batch !! gradient_accum !! GPUs !! Effective Batch
|-
| 7B model, 24GB GPU || 1 || 32 || 1 || 32
|-
| 7B model, 2x 24GB || 2 || 8 || 2 || 32
|-
| 13B model, 80GB || 4 || 8 || 1 || 32
|-
| 70B model, 8x 80GB || 1 || 8 || 8 || 64
|}

== Related Pages ==
* [[uses_heuristic::Implementation:huggingface_transformers_TrainingArguments_setup]]
* [[uses_heuristic::Implementation:huggingface_transformers_Training_execution]]
* [[uses_heuristic::Workflow:huggingface_transformers_Model_Training_Trainer]]
