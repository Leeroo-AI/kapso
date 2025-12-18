{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|PEFT Docs|https://huggingface.co/docs/peft/conceptual_guides/lora#merge-lora-weights-into-the-base-model]]
|-
! Domains
| [[domain::Adapter]], [[domain::Inference]], [[domain::Deployment]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Concrete tool for merging LoRA adapter weights into the base model and removing the PEFT wrapper for deployment.

=== Description ===

`merge_and_unload` permanently merges the adapter weights into the base model weights and removes the PEFT infrastructure. The result is a standard transformers model with the adapter effects baked in. This enables deployment without PEFT dependencies and may provide faster inference.

=== Usage ===

Use this when deploying a trained model where you don't need adapter switching. The operation is irreversible on the model instance, so assign the result to a new variable. Use `safe_merge=True` to detect potential numerical issues.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft peft]
* '''File:''' src/peft/tuners/tuners_utils.py
* '''Lines:''' L611-647

=== Signature ===
<syntaxhighlight lang="python">
def merge_and_unload(
    self,
    progressbar: bool = False,
    safe_merge: bool = False,
    adapter_names: Optional[list[str]] = None
) -> torch.nn.Module:
    """
    Merge adapter weights into base model and remove PEFT wrapper.

    Args:
        progressbar: Show progress bar during merge
        safe_merge: Check for NaNs during merge (slower but safer)
        adapter_names: Specific adapters to merge. None = all active

    Returns:
        Base model with merged weights (no PEFT wrapper)
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Method on PeftModel, no explicit import
# merged_model = model.merge_and_unload()
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| progressbar || bool || No || Show tqdm progress. Default: False
|-
| safe_merge || bool || No || Check for NaNs (slower). Default: False
|-
| adapter_names || list[str] || No || Adapters to merge. Default: all active
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || torch.nn.Module || Standard transformers model with merged weights
|}

== Usage Examples ==

=== Basic Merge for Deployment ===
<syntaxhighlight lang="python">
from peft import PeftModel

# Load adapter
model = PeftModel.from_pretrained(base_model, "path/to/adapter")

# Merge and get standard model
merged_model = model.merge_and_unload()

# Save as standard transformers model
merged_model.save_pretrained("./merged-model")
</syntaxhighlight>

=== Safe Merge with Progress ===
<syntaxhighlight lang="python">
# Merge with safety checks
merged_model = model.merge_and_unload(
    progressbar=True,
    safe_merge=True,  # Slower but detects NaN issues
)
</syntaxhighlight>

=== Merge Specific Adapters ===
<syntaxhighlight lang="python">
# Model with multiple adapters
model.load_adapter("adapter1", adapter_name="math")
model.load_adapter("adapter2", adapter_name="code")

# Merge only the math adapter
model.set_adapter("math")
merged_model = model.merge_and_unload(adapter_names=["math"])
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_peft_Adapter_Merging_Into_Base]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
