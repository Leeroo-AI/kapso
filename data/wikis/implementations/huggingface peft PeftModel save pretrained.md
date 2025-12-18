{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|PEFT Docs|https://huggingface.co/docs/peft/quicktour#save-and-load-a-model]]
|-
! Domains
| [[domain::Serialization]], [[domain::Fine_Tuning]], [[domain::Adapter]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Concrete tool for saving trained LoRA adapter weights and configuration to disk for later loading.

=== Description ===

`PeftModel.save_pretrained()` saves only the trained adapter weights and configuration, not the base model. This results in very small checkpoint sizes (typically 10-100MB vs. multiple GB for full models). The saved adapter can be loaded onto any compatible base model using `PeftModel.from_pretrained()`.

=== Usage ===

Call this after training completes to persist your adapter. The function creates `adapter_model.safetensors` (weights) and `adapter_config.json` (configuration). For multi-adapter models, use `selected_adapters` to save specific adapters. Enable `safe_serialization=True` (default) for safetensors format.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft peft]
* '''File:''' src/peft/peft_model.py
* '''Lines:''' L190-386

=== Signature ===
<syntaxhighlight lang="python">
def save_pretrained(
    self,
    save_directory: str,
    safe_serialization: bool = True,
    selected_adapters: Optional[list[str]] = None,
    save_embedding_layers: Union[str, bool] = "auto",
    is_main_process: bool = True,
    path_initial_model_for_weight_conversion: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """
    Save adapter weights and configuration.

    Args:
        save_directory: Output directory path
        safe_serialization: Use safetensors format (recommended)
        selected_adapters: List of adapter names to save (None = all)
        save_embedding_layers: Save modified embeddings ("auto" checks config)
        is_main_process: Only save on main process (for distributed)
        path_initial_model_for_weight_conversion: For PiSSA/CorDA conversion

    Creates:
        - adapter_model.safetensors (or adapter_model.bin)
        - adapter_config.json
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Method on PeftModel, no explicit import needed
# model.save_pretrained(...)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| save_directory || str || Yes || Output directory path (created if not exists)
|-
| safe_serialization || bool || No || Use safetensors format. Default: True
|-
| selected_adapters || list[str] || No || Specific adapters to save. Default: all
|-
| save_embedding_layers || str or bool || No || Save embeddings. "auto" checks target_modules
|-
| is_main_process || bool || No || Only save on main process (DDP). Default: True
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| adapter_model.safetensors || File || Adapter weights in safetensors format
|-
| adapter_config.json || File || LoraConfig serialized as JSON
|-
| README.md || File || Model card (if created)
|}

== Usage Examples ==

=== Basic Save After Training ===
<syntaxhighlight lang="python">
from peft import get_peft_model, LoraConfig

# After training...
trainer.train()

# Save the adapter
model.save_pretrained("./my-lora-adapter")

# Directory contents:
# ./my-lora-adapter/
#   adapter_model.safetensors  (~40MB for 7B model with r=16)
#   adapter_config.json        (~1KB)
</syntaxhighlight>

=== Save Specific Adapters ===
<syntaxhighlight lang="python">
# Model with multiple adapters
model.load_adapter("./math-adapter", adapter_name="math")
model.load_adapter("./code-adapter", adapter_name="code")

# Save only the math adapter
model.save_pretrained(
    "./saved-adapters",
    selected_adapters=["math"],
)

# Save all adapters to subdirectories
model.save_pretrained(
    "./saved-adapters",
    selected_adapters=["math", "code"],
)
# Creates:
#   ./saved-adapters/math/adapter_model.safetensors
#   ./saved-adapters/code/adapter_model.safetensors
</syntaxhighlight>

=== Push to HuggingFace Hub ===
<syntaxhighlight lang="python">
# Save locally then push
model.save_pretrained("./my-adapter")
model.push_to_hub("username/my-lora-adapter")

# Or push directly
model.push_to_hub(
    "username/my-lora-adapter",
    private=True,
    commit_message="Add trained LoRA adapter",
)
</syntaxhighlight>

=== With PiSSA/CorDA Conversion ===
<syntaxhighlight lang="python">
# Convert PiSSA adapter to standard LoRA for compatibility
model.save_pretrained(
    "./converted-adapter",
    path_initial_model_for_weight_conversion="./initial-pissa-adapter",
)
# The saved adapter can now be used without PiSSA-specific loading
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_peft_Adapter_Serialization]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
