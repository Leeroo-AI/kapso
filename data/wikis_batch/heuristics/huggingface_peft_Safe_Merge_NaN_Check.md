# Heuristic: huggingface_peft_Safe_Merge_NaN_Check

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace PEFT|https://github.com/huggingface/peft]]
|-
! Domains
| [[domain::Inference]], [[domain::Debugging]], [[domain::Quality_Assurance]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==
Use `safe_merge=True` when merging adapters to detect broken adapters that produce NaN values in merged weights.

=== Description ===
The `safe_merge` parameter in `merge_and_unload()` performs a copy of the original weights before merging and validates that the resulting weights contain no NaN or Inf values. This is useful for catching broken adapters or numerical instabilities before deployment.

=== Usage ===
Enable `safe_merge=True` when:
- Deploying merged models to production
- Debugging unexpected model behavior
- Merging adapters trained with experimental settings
- Combining multiple adapters via `add_weighted_adapter`

== The Insight (Rule of Thumb) ==

* **Action:** Call `model.merge_and_unload(safe_merge=True)` for production deployments
* **Value:** Boolean flag, default is False
* **Trade-off:** Additional memory for weight copy during merge; slight overhead for NaN check
* **Benefit:** Early detection of broken adapters prevents silent failures in production

== Reasoning ==

Adapters can become corrupted due to:
- Training instabilities (gradient explosion, learning rate too high)
- Checkpoint corruption
- Incompatible dtype conversions
- Bugs in custom training code

The safe merge creates a copy of weights, performs the merge on the copy, validates for finite values, and only then commits the change. This prevents corrupting the original model weights.

== Code Evidence ==

Safe merge validation from `src/peft/tuners/lora/bnb.py:128-132` (4-bit):
<syntaxhighlight lang="python">
if safe_merge and not torch.isfinite(w_data).all():
    raise ValueError(
        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
    )
</syntaxhighlight>

Same pattern for 8-bit from `src/peft/tuners/lora/bnb.py:128-132`:
<syntaxhighlight lang="python">
if safe_merge and not torch.isfinite(w_data).all():
    raise ValueError(
        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
    )
</syntaxhighlight>

Bias check from `src/peft/tuners/lora/bnb.py:139-142`:
<syntaxhighlight lang="python">
if self.lora_bias[active_adapter]:
    bias_data = self.get_base_layer().bias.data + self.lora_B[active_adapter].bias
    if safe_merge and not torch.isfinite(bias_data):
        raise ValueError(
            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
        )
</syntaxhighlight>

== Related Pages ==
* [[uses_heuristic::Implementation:huggingface_peft_merge_and_unload]]
* [[uses_heuristic::Workflow:huggingface_peft_Adapter_Inference]]
* [[uses_heuristic::Principle:huggingface_peft_Adapter_Merging]]

[[Category:Heuristic]]
[[Category:Debugging]]
[[Category:Quality_Assurance]]
[[Category:Inference]]
