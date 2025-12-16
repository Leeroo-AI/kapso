# Implementation: unsloth_save_model

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Documentation|https://docs.unsloth.ai]]
|-
! Domains
| [[domain::Model_Export]], [[domain::Fine_Tuning]], [[domain::Deployment]]
|-
! Last Updated
| [[last_updated::2025-12-15 19:00 GMT]]
|}

== Overview ==
Concrete tool for saving fine-tuned models with LoRA merging and various export options provided by the Unsloth library.

=== Description ===
`unsloth_save_model` is the core function for saving trained Unsloth models. It handles:

1. **LoRA Saving** - Save just the LoRA adapter weights (smallest, fastest)
2. **16-bit Merging** - Merge LoRA weights into base model and save in float16
3. **4-bit Merging** - Merge and save in 4-bit format for continued training
4. **Hub Pushing** - Direct upload to HuggingFace Hub with proper metadata

The function automatically handles:
- Memory-efficient weight dequantization from 4-bit to 16-bit
- Sharded saving for large models
- Disk offloading when VRAM/RAM is limited
- Tokenizer saving with inference-ready padding configuration

This function is typically accessed through the model's `save_pretrained_merged()` method, which Unsloth patches onto models.

=== Usage ===
Use this function when you need to:
- Save a fine-tuned model after training
- Merge LoRA adapters into base weights
- Prepare models for GGUF conversion (requires merged 16-bit)
- Upload trained models to HuggingFace Hub

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unslothai/unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/save.py#L228-L851 unsloth/save.py]
* '''Lines:''' 228-851

Source Files: unsloth/save.py:L228-L851

=== Signature ===
<syntaxhighlight lang="python">
@torch.inference_mode
def unsloth_save_model(
    model: Union[PreTrainedModel, PeftModel],
    tokenizer: PreTrainedTokenizer,
    save_directory: Union[str, os.PathLike],
    save_method: str = "lora",  # ["lora", "merged_16bit", "merged_4bit"]
    push_to_hub: bool = False,
    token: Optional[Union[str, bool]] = None,
    is_main_process: bool = True,
    state_dict: Optional[dict] = None,
    save_function: Callable = torch.save,
    max_shard_size: Union[int, str] = "5GB",
    safe_serialization: bool = True,
    variant: Optional[str] = None,
    save_peft_format: bool = True,
    # Push to hub options
    use_temp_dir: Optional[bool] = None,
    commit_message: Optional[str] = "Trained with Unsloth",
    private: Optional[bool] = None,
    create_pr: bool = False,
    revision: str = None,
    commit_description: str = "Upload model trained with Unsloth 2x faster",
    tags: List[str] = None,
    # Unsloth options
    temporary_location: str = "_unsloth_temporary_saved_buffers",
    maximum_memory_usage: float = 0.9,
) -> Tuple[str, Optional[str]]:
    """
    Save a fine-tuned model with optional LoRA merging and Hub upload.

    Args:
        model: The trained model (PeftModel or base model)
        tokenizer: Associated tokenizer
        save_directory: Local path or HuggingFace repo name
        save_method: One of:
            - "lora": Save only LoRA adapters (fastest, smallest)
            - "merged_16bit": Merge LoRA and save full model in float16
            - "merged_4bit": Merge and keep 4-bit quantization
        push_to_hub: Whether to upload to HuggingFace Hub
        token: HuggingFace API token for Hub upload
        max_shard_size: Maximum size per shard file (e.g., "5GB")
        safe_serialization: Use safetensors format
        commit_message: Git commit message for Hub
        private: Make Hub repository private
        tags: Additional tags for the model
        temporary_location: Disk location for overflow during merge
        maximum_memory_usage: Max GPU/RAM fraction to use (0.0-0.95)

    Returns:
        Tuple of (save_directory, username) - username is None for local saves
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth.save import unsloth_save_model

# Or use the patched model method:
model.save_pretrained_merged(...)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || PreTrainedModel/PeftModel || Yes || Trained model to save
|-
| tokenizer || PreTrainedTokenizer || Yes || Tokenizer to save alongside model
|-
| save_directory || str || Yes || Local path or HuggingFace repo ID
|-
| save_method || str || No || "lora", "merged_16bit", or "merged_4bit"
|-
| push_to_hub || bool || No || Upload to HuggingFace Hub (default: False)
|-
| token || str || No || HF API token (required if push_to_hub=True)
|-
| max_shard_size || str/int || No || Max shard size (default: "5GB")
|-
| safe_serialization || bool || No || Use safetensors (default: True)
|-
| private || bool || No || Make Hub repo private
|-
| maximum_memory_usage || float || No || GPU/RAM usage limit (default: 0.9)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| save_directory || str || Path/repo where model was saved
|-
| username || str/None || HuggingFace username (None for local)
|-
| Files || Various || model.safetensors, config.json, tokenizer files
|}

== Usage Examples ==

=== Save LoRA Adapter Only ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# After training...

# Save just the LoRA adapter (smallest, fastest)
model.save_pretrained("lora_adapter")
tokenizer.save_pretrained("lora_adapter")

# This creates:
# - adapter_config.json
# - adapter_model.safetensors
# - tokenizer files
</syntaxhighlight>

=== Save Merged 16-bit Model ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# After training...

# Merge LoRA into base model and save as 16-bit
# Required before GGUF conversion
model.save_pretrained_merged(
    "merged_model_16bit",
    tokenizer,
    save_method="merged_16bit",
)

# This creates a complete HuggingFace model directory:
# - model.safetensors (or sharded model-00001-of-00002.safetensors)
# - config.json
# - tokenizer files
</syntaxhighlight>

=== Push to HuggingFace Hub ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# After training...

# Push merged model directly to Hub
model.push_to_hub_merged(
    "username/my-finetuned-model",
    tokenizer,
    save_method="merged_16bit",
    token="hf_xxxx",
    private=False,
    commit_message="Fine-tuned Llama on custom dataset",
)

# Model available at: https://huggingface.co/username/my-finetuned-model
</syntaxhighlight>

=== Save with Custom Sharding ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# For large models, control shard size
model.save_pretrained_merged(
    "large_model_output",
    tokenizer,
    save_method="merged_16bit",
    max_shard_size="2GB",  # Smaller shards for easier upload
    safe_serialization=True,
)
</syntaxhighlight>

=== Direct Function Usage ===
<syntaxhighlight lang="python">
from unsloth.save import unsloth_save_model

# Direct function call (same as model.save_pretrained_merged)
save_dir, username = unsloth_save_model(
    model=model,
    tokenizer=tokenizer,
    save_directory="output_dir",
    save_method="merged_16bit",
    push_to_hub=True,
    token="hf_xxxx",
    maximum_memory_usage=0.8,  # Use 80% of available memory
)

print(f"Saved to: {save_dir}")
if username:
    print(f"Uploaded by: {username}")
</syntaxhighlight>

=== Memory-Constrained Saving ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# For environments with limited memory (Colab, Kaggle)
model.save_pretrained_merged(
    "merged_model",
    tokenizer,
    save_method="merged_16bit",
    maximum_memory_usage=0.7,  # Conservative memory usage
    temporary_location="/tmp/unsloth_save",  # Use tmp for overflow
)
</syntaxhighlight>

== Related Pages ==
=== Context & Requirements ===
* Requires sufficient disk space for merged models
* Hub upload requires valid HuggingFace token
* merged_16bit required before GGUF conversion

=== Tips and Tricks ===
* Use "lora" save_method for iterative training (fastest, smallest)
* Use "merged_16bit" before GGUF conversion
* Avoid "merged_4bit" unless continuing 4-bit training
* Lower maximum_memory_usage on memory-constrained systems
