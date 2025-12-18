# Heuristic: huggingface_peft_Gradient_Checkpointing

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|HuggingFace Transformers|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::Optimization]], [[domain::Memory]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2024-12-18 00:00 GMT]]
|}

== Overview ==
Memory optimization technique that trades compute for VRAM by recomputing activations during backprop.

=== Description ===
Gradient checkpointing (activation checkpointing) reduces peak memory usage during training by not storing all intermediate activations. Instead, activations are recomputed during the backward pass. This is especially important for QLoRA training where models barely fit in VRAM.

=== Usage ===
Use this heuristic when:
- Training large models (7B+ parameters) on limited VRAM
- Running out of GPU memory during training
- Using `prepare_model_for_kbit_training()` (enabled by default)
- Need to increase batch size but memory is constrained

== The Insight (Rule of Thumb) ==

* **Action:** Enable gradient checkpointing via:
  1. `prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)` (default)
  2. `model.gradient_checkpointing_enable()` directly
  3. `TrainingArguments(gradient_checkpointing=True)` in Trainer

* **Value:** Boolean flag - enable when VRAM is constrained

* **Memory Savings:** Typically 50-60% reduction in activation memory

* **Trade-off:**
  * ~20-30% slower training due to recomputation
  * Requires slightly more compute per step
  * Allows significantly larger batch sizes or models

* **Note:** Must disable KV cache (`use_cache=False`) when gradient checkpointing is enabled

* **Gradient Checkpointing Kwargs:**
  * Can pass custom kwargs via `gradient_checkpointing_kwargs`
  * Example: `{"use_reentrant": False}` for newer PyTorch behavior

== Reasoning ==

### Memory-Compute Trade-off
Deep transformers store activation tensors at each layer for backpropagation. For a model with L layers:
- Without checkpointing: Store ~L activation tensors
- With checkpointing: Store ~sqrt(L) checkpoints, recompute others

### Why It Matters for QLoRA
QLoRA already pushes memory limits by using 4-bit quantization. Gradient checkpointing provides additional headroom to:
- Use larger batch sizes for better gradient estimates
- Train longer sequences
- Fit models that would otherwise OOM

### KV Cache Conflict
Gradient checkpointing recomputes forward passes, but KV cache assumes stored key-value pairs. These are incompatible, so `use_cache` must be False during training.

== Code Evidence ==

Default enabled in `prepare_model_for_kbit_training` from `other.py:130-140`:
<syntaxhighlight lang="python">
def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=None):
    """
    This method wraps the entire protocol for preparing a model before running a training.
    This includes:
        1- Cast the layernorm in fp32
        2- making output embedding layer require grads
        3- Add the upcasting of the lm head to fp32
        4- Freezing the base model layers to ensure they are not updated during training
    """
</syntaxhighlight>

Gradient checkpointing enablement from `other.py:195-207`:
<syntaxhighlight lang="python">
if use_gradient_checkpointing:
    # For backward compatibility
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # To support older transformers versions, check if the model supports gradient_checkpointing_kwargs
    _supports_gc_kwargs = "gradient_checkpointing_kwargs" in list(
        inspect.signature(model.gradient_checkpointing_enable).parameters
    )
</syntaxhighlight>

use_cache conflict handling from `peft_model.py:2099-2100`:
<syntaxhighlight lang="python">
# TODO: starting with transformers 4.38, all architectures should support caching.
# We can remove the hasattr check once we drop support for older transformers versions.
</syntaxhighlight>

== Related Pages ==
* [[uses_heuristic::Implementation:huggingface_peft_prepare_model_for_kbit_training]]
* [[uses_heuristic::Workflow:huggingface_peft_QLoRA_Training]]
* [[uses_heuristic::Workflow:huggingface_peft_LoRA_Fine_Tuning]]
