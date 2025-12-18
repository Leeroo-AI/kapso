{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|LoRA|https://arxiv.org/abs/2106.09685]]
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
|-
! Domains
| [[domain::Adapter]], [[domain::Deployment]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Principle for permanently merging adapter weights into the base model to create a standalone deployable model.

=== Description ===

Adapter Merging combines the low-rank adapter matrices with the frozen base weights to produce a standard model. The resulting model:
* Has no PEFT dependencies
* Runs with standard inference code
* May have faster inference (no adapter computation)
* Cannot switch adapters (irreversible)

=== Usage ===

Apply this when deploying a model where:
* Adapter switching is not needed
* PEFT dependency should be removed
* Maximum inference speed is required
* Model will be shared without PEFT context

== Theoretical Basis ==

'''Weight Merging:'''

The core merge operation:
<math>W_{merged} = W_0 + \frac{\alpha}{r} \cdot BA</math>

Where:
* <math>W_0</math> is the original frozen weight
* <math>\frac{\alpha}{r}</math> is the LoRA scaling factor
* <math>BA</math> is the learned low-rank update

'''Merge Implementation:'''

<syntaxhighlight lang="python">
# Pseudo-code for merge operation
def merge_lora_weights(layer):
    delta_w = (layer.lora_B.weight @ layer.lora_A.weight) * layer.scaling
    layer.weight.data += delta_w

    # Remove LoRA components
    del layer.lora_A
    del layer.lora_B
</syntaxhighlight>

'''Safe Merge:'''

Safe merge checks for numerical issues:
<syntaxhighlight lang="python">
def safe_merge(layer):
    delta_w = layer.lora_B.weight @ layer.lora_A.weight

    # Check for NaNs or Infs
    if torch.isnan(delta_w).any() or torch.isinf(delta_w).any():
        raise ValueError("NaN/Inf detected in adapter weights")

    layer.weight.data += delta_w * layer.scaling
</syntaxhighlight>

'''Unload Process:'''

After merging, the PEFT wrapper is removed:
<syntaxhighlight lang="python">
# Replace LoRA layer with merged Linear
for parent, name, module in model.named_modules():
    if isinstance(module, LoraLayer):
        merged_linear = module.get_base_layer()  # Already merged
        setattr(parent, name, merged_linear)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_peft_merge_and_unload]]
