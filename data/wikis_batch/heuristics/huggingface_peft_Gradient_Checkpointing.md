# Heuristic: huggingface_peft_Gradient_Checkpointing

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|PyTorch Checkpointing|https://pytorch.org/docs/stable/checkpoint.html]]
|-
! Domains
| [[domain::Memory_Optimization]], [[domain::Training]], [[domain::LLMs]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==
Memory optimization technique using gradient checkpointing to reduce VRAM usage during quantized model training.

=== Description ===
Gradient checkpointing (activation checkpointing) trades compute for memory by not storing intermediate activations during the forward pass. Instead, activations are recomputed during the backward pass. This is automatically enabled by `prepare_model_for_kbit_training()` and is essential for training large models on consumer GPUs.

=== Usage ===
Use this heuristic when you are **VRAM constrained** and training with quantized models (QLoRA). It is enabled by default in `prepare_model_for_kbit_training()` and is standard practice for fine-tuning 7B+ parameter models on consumer hardware.

== The Insight (Rule of Thumb) ==

* **Action:** Enable gradient checkpointing via `prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)`
* **Value:** Enabled by default (True)
* **Trade-off:** Reduces VRAM by 30-50% at cost of ~20% slower training
* **Compatibility:** Requires `use_reentrant=False` recommended for newer PyTorch versions

== Reasoning ==

Deep transformers have massive activation maps (Batch × SeqLen × Hidden). Storing all activations for backpropagation is the primary VRAM bottleneck. By recomputing activations during backward pass, we can fit larger batch sizes or train larger models on limited hardware.

The `prepare_model_for_kbit_training()` function also:
1. Casts layer normalization to fp32 for stability
2. Freezes base model parameters
3. Enables input gradients for backward compatibility

== Code Evidence ==

From `src/peft/utils/other.py:130-215`:
<syntaxhighlight lang="python">
def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=None):
    """
    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32 4- Freezing the base model layers to ensure they are not updated during training
    """
    # ... freeze base model layers ...

    if loaded_in_kbit and use_gradient_checkpointing:
        # When having `use_reentrant=False` + gradient_checkpointing, there is no need for this hack
        if "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]:
            # For backward compatibility
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable(**gc_enable_kwargs)
    return model
</syntaxhighlight>

Warning for older transformers from `src/peft/utils/other.py:202-207`:
<syntaxhighlight lang="python">
if not _supports_gc_kwargs and len(gradient_checkpointing_kwargs) > 0:
    warnings.warn(
        "gradient_checkpointing_kwargs is not supported in this version of transformers. The passed kwargs will be ignored."
        " if you want to use that feature, please upgrade to the latest version of transformers.",
        FutureWarning,
    )
</syntaxhighlight>

== Related Pages ==
* [[uses_heuristic::Implementation:huggingface_peft_prepare_model_for_kbit_training]]
* [[uses_heuristic::Workflow:huggingface_peft_QLoRA_Training]]
* [[uses_heuristic::Principle:huggingface_peft_Memory_Optimization]]

[[Category:Heuristic]]
[[Category:Memory_Optimization]]
[[Category:Training]]
