# Heuristic: huggingface_transformers_Gradient_Checkpointing

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Performance Training|https://huggingface.co/docs/transformers/perf_train_gpu_one]]
|-
! Domains
| [[domain::Optimization]], [[domain::Memory]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==
Memory optimization technique using Gradient Checkpointing to reduce VRAM usage by 50-60% at the cost of ~20% slower training.

=== Description ===
Gradient Checkpointing (activation checkpointing) drastically reduces memory usage during training. Instead of storing all intermediate activations for the backward pass, it stores only a subset and recomputes the rest on-the-fly. This effectively trades a small increase in computation time (20-30%) for a massive reduction in peak memory usage (up to 50-60%). This is essential for training large models on consumer GPUs.

=== Usage ===
Use this heuristic when you are **VRAM constrained** (e.g., getting CUDA OOM errors) or need to fit a model that is too large for your GPU memory. It is standard practice when fine-tuning 7B+ parameter models on consumer hardware (e.g., RTX 3090/4090 with 24GB VRAM).

== The Insight (Rule of Thumb) ==

* **Action:** Enable gradient checkpointing via `model.gradient_checkpointing_enable()` or `TrainingArguments(gradient_checkpointing=True)`
* **Value:** N/A (Boolean flag)
* **Trade-off:** Reduces VRAM usage by ~50-60% at the cost of ~20-30% slower training speed
* **Compatibility:** Requires `use_cache=False` during training (automatic with Trainer)

== Reasoning ==

Deep Transformers have massive activation maps (Batch x SeqLen x Hidden). Storing these for backpropagation is the primary VRAM bottleneck. Recomputing them is compute-bound but allows fitting significantly larger batch sizes or longer sequences.

Benchmarks on Llama-2-7B show:
- Without checkpointing: ~22GB VRAM for batch_size=1, seq_len=2048
- With checkpointing: ~11GB VRAM for same configuration
- Training time increase: ~25%

== Code Evidence ==

From `trainer.py:L1989-1992`:

<syntaxhighlight lang="python">
if self.args.gradient_checkpointing:
    if model.config.use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. "
            "Setting `use_cache=False`."
        )
</syntaxhighlight>

Usage in TrainingArguments from `training_args.py`:

<syntaxhighlight lang="python">
gradient_checkpointing (`bool`, *optional*, defaults to `False`):
    If True, use gradient checkpointing to save memory at the expense of
    slower backward pass.
</syntaxhighlight>

Model enablement from `modeling_utils.py`:

<syntaxhighlight lang="python">
def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
    """
    Activates gradient checkpointing for the current model.
    """
    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {"use_reentrant": False}

    self._set_gradient_checkpointing(enable=True,
                                      gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
</syntaxhighlight>

== Example Usage ==

<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments

# Method 1: Via TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    gradient_checkpointing=True,  # Enable gradient checkpointing
    per_device_train_batch_size=4,
    # ... other args
)

# Method 2: Direct model call
model.gradient_checkpointing_enable()

# For custom training loop, disable use_cache
model.config.use_cache = False
</syntaxhighlight>

== Related Pages ==
* [[uses_heuristic::Implementation:huggingface_transformers_Training_execution]]
* [[uses_heuristic::Implementation:huggingface_transformers_TrainingArguments_setup]]
* [[uses_heuristic::Workflow:huggingface_transformers_Model_Training_Trainer]]
