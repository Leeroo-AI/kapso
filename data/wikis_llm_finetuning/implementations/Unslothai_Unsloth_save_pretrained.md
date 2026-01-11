# Implementation: save_pretrained

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|HuggingFace Model Saving|https://huggingface.co/docs/transformers/main_classes/model]]
|-
! Domains
| [[domain::NLP]], [[domain::Model_Serialization]], [[domain::Deployment]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Concrete tool for saving fine-tuned models in various formats (LoRA adapters, merged 16-bit, merged 4-bit) for deployment or further training, provided by Unsloth.

=== Description ===

`model.save_pretrained` and `model.save_pretrained_merged` provide flexible model saving options after QLoRA fine-tuning:

* **LoRA only** (`save_method="lora"`): Saves only the trained adapter weights (~50-100MB). Fastest, smallest, requires base model at inference.
* **Merged 16-bit** (`save_method="merged_16bit"`): Merges LoRA into base and saves as float16. Required for GGUF export.
* **Merged 4-bit** (`save_method="merged_4bit"`): Merges LoRA into quantized weights. For direct inference without base model.

=== Usage ===

Call after training completes. Choose save method based on deployment needs:
* For continued training or HuggingFace deployment: `lora`
* For llama.cpp/GGUF export: `merged_16bit`
* For direct 4-bit inference: `merged_4bit`

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/save.py
* '''Lines:''' L235-860

=== Signature ===
<syntaxhighlight lang="python">
def unsloth_save_model(
    model,
    tokenizer,
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
    # Push to hub
    use_temp_dir: Optional[bool] = None,
    commit_message: Optional[str] = "Trained with Unsloth",
    private: Optional[bool] = None,
    create_pr: bool = False,
    revision: str = None,
    commit_description: str = "Upload model trained with Unsloth 2x faster",
    tags: List[str] = None,
    # Unsloth-specific
    temporary_location: str = "_unsloth_temporary_saved_buffers",
    maximum_memory_usage: float = 0.9,
) -> None:
    """
    Save model with optional LoRA merging.

    Args:
        model: Trained model with LoRA adapters
        tokenizer: Tokenizer to save alongside model
        save_directory: Output path or HuggingFace repo ID
        save_method: "lora", "merged_16bit", or "merged_4bit"
        push_to_hub: Upload to HuggingFace Hub
        token: HuggingFace API token
        max_shard_size: Maximum file size per shard
        safe_serialization: Use safetensors format
        maximum_memory_usage: RAM limit for merging (0.0-0.95)

    Returns:
        None (files saved to disk or uploaded to Hub)
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# save_pretrained is a method on the model
model.save_pretrained("./output")
model.save_pretrained_merged("./output", tokenizer, save_method="merged_16bit")
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| save_directory || str || Yes || Local path or HuggingFace repo ID
|-
| tokenizer || PreTrainedTokenizer || Yes (for merged) || Tokenizer to save with model
|-
| save_method || str || No (default: "lora") || "lora", "merged_16bit", or "merged_4bit"
|-
| push_to_hub || bool || No (default: False) || Upload to HuggingFace Hub
|-
| token || str || Conditional || Required if push_to_hub=True
|-
| max_shard_size || str || No (default: "5GB") || Maximum size per weight shard
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Files || Directory || Model weights, config, tokenizer files saved to disk
|-
| Hub URL || str || Repository URL if push_to_hub=True
|}

== Usage Examples ==

=== Save LoRA Adapters Only ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# After training...
# Save just the LoRA adapters (smallest, fastest)
model.save_pretrained("./lora_model")
tokenizer.save_pretrained("./lora_model")

# Load later with:
# model, tokenizer = FastLanguageModel.from_pretrained("./lora_model")
</syntaxhighlight>

=== Save Merged 16-bit for GGUF Export ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# After training...
# Merge LoRA and save as float16 (required for GGUF)
model.save_pretrained_merged(
    "./merged_model",
    tokenizer,
    save_method = "merged_16bit",
    max_shard_size = "5GB",
)

# Now ready for: model.save_pretrained_gguf(...)
</syntaxhighlight>

=== Push to HuggingFace Hub ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# After training...
# Upload LoRA adapters to Hub
model.save_pretrained_merged(
    "username/my-fine-tuned-model",  # Repo ID
    tokenizer,
    save_method = "lora",
    push_to_hub = True,
    token = "hf_your_token",
    private = False,
)
</syntaxhighlight>

=== Save Merged for Direct Inference ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# After training...
# Merge LoRA into 4-bit weights (smallest merged)
model.save_pretrained_merged(
    "./merged_4bit",
    tokenizer,
    save_method = "merged_4bit_forced",  # Acknowledge accuracy warning
    max_shard_size = "2GB",
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_Model_Saving]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
