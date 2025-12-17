# Heuristic: huggingface_peft_DoRA_Mixed_Batch_Limitation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|DoRA Paper|https://arxiv.org/abs/2402.09353]]
|-
! Domains
| [[domain::DoRA]], [[domain::Multi_Adapter]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==
DoRA (Weight-Decomposed Low-Rank Adaptation) is incompatible with mixed-batch adapter inference using `adapter_names`.

=== Description ===
When using DoRA adapters, you cannot pass the `adapter_names` argument to mix different adapters within the same batch. This is a fundamental limitation of DoRA's weight decomposition approach, which applies a magnitude normalization that doesn't support per-sample adapter selection.

=== Usage ===
If you need mixed-batch multi-adapter inference (different adapters for different samples in the same forward pass), use standard LoRA instead of DoRA. DoRA adapters can still be used for single-adapter or switched-adapter inference.

== The Insight (Rule of Thumb) ==

* **Action:** Do not use `adapter_names` argument with DoRA-enabled adapters
* **Constraint:** Mixed-batch inference only works with vanilla LoRA
* **Trade-off:** DoRA offers better quality but loses mixed-batch flexibility
* **Alternative:** Use `model.set_adapter()` to switch adapters between batches

== Reasoning ==

DoRA decomposes weight updates into magnitude and direction components. The magnitude normalization is computed per-layer, not per-sample, making it impossible to efficiently apply different DoRA adapters to different samples in the same batch without significant architectural changes.

The `adapter_names` feature was designed for vanilla LoRA where the adapter contribution is a simple additive delta that can be computed independently per sample.

== Code Evidence ==

Validation check from `src/peft/tuners/lora/layer.py:540-544`:
<syntaxhighlight lang="python">
unique_adapters = {name for name in adapter_names if name != "__base__"}
for adapter_name in unique_adapters:
    if self.use_dora.get(adapter_name, False):
        msg = "Cannot pass `adapter_names` when DoRA is enabled."
        raise ValueError(msg)
</syntaxhighlight>

== Related Pages ==
* [[uses_heuristic::Implementation:huggingface_peft_LoraConfig]]
* [[uses_heuristic::Workflow:huggingface_peft_Multi_Adapter_Management]]

[[Category:Heuristic]]
[[Category:DoRA]]
[[Category:Multi_Adapter]]
[[Category:Inference]]
