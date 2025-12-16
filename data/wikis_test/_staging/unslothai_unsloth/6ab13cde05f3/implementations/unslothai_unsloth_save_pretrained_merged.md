{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Save Guide|https://docs.unsloth.ai/basics/running-and-saving-models]]
|-
! Domains
| [[domain::Model_Export]], [[domain::Fine_Tuning]], [[domain::LoRA]], [[domain::Deployment]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==

Concrete tool for merging LoRA adapter weights into base model weights and saving the result in HuggingFace-compatible format, provided by the Unsloth library.

=== Description ===

`save_pretrained_merged` is a method dynamically attached to models by Unsloth that handles the process of:

* **LoRA weight merging**: Combines LoRA adapter matrices (B×A) with base weights
* **Dequantization**: Converts 4-bit quantized weights back to 16-bit precision
* **Efficient saving**: Uses memory-efficient chunked saving to avoid OOM on large models
* **Format compatibility**: Outputs standard HuggingFace safetensors format

The method performs the merge operation: W_merged = W_base + α×(B×A) where:
- W_base: Original quantized weights (dequantized to float32)
- B, A: LoRA adapter matrices
- α: LoRA scaling factor

This produces a self-contained model that can be:
- Loaded without PEFT library
- Converted to GGUF for llama.cpp
- Deployed with vLLM, SGLang, or TGI

=== Usage ===

Call `model.save_pretrained_merged()` after training to create a deployment-ready model. This is typically the final step before GGUF conversion or Hub upload.

Use cases:
* Preparing a model for GGUF conversion (llama.cpp, Ollama)
* Creating a standalone model without LoRA dependencies
* Uploading a merged model to HuggingFace Hub

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unslothai_unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/save.py unsloth/save.py]
* '''Lines:''' 228-506 (unsloth_save_model), 2996-3068 (patch_saving_functions)

=== Signature ===
<syntaxhighlight lang="python">
def save_pretrained_merged(
    self,
    save_directory: Union[str, os.PathLike],
    tokenizer = None,
    save_method: str = "merged_16bit",  # "lora", "merged_16bit", "merged_4bit"
    push_to_hub: bool = False,
    token: Optional[str] = None,
    is_main_process: bool = True,
    state_dict: Optional[dict] = None,
    save_function: Callable = torch.save,
    max_shard_size: Union[int, str] = "5GB",
    safe_serialization: bool = True,
    variant: Optional[str] = None,
    save_peft_format: bool = True,
    # Push to hub settings
    private: Optional[bool] = None,
    commit_message: Optional[str] = "Trained with Unsloth",
    # Unsloth-specific
    temporary_location: str = "_unsloth_temporary_saved_buffers",
    maximum_memory_usage: float = 0.9,
):
    """
    Merge LoRA weights and save to HuggingFace format.

    Args:
        save_directory: Output directory path
        tokenizer: Tokenizer to save alongside model
        save_method: "lora" (adapters only), "merged_16bit" (full merged),
                    or "merged_4bit" (merged + requantized)
        push_to_hub: Whether to upload to HuggingFace Hub
        token: HuggingFace token for Hub upload
        max_shard_size: Maximum size per shard file (e.g., "5GB")
        safe_serialization: Use safetensors format (recommended)
        maximum_memory_usage: GPU memory threshold for chunked saving
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
# save_pretrained_merged is a method attached to the model
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| save_directory || str || Yes || Output directory or HuggingFace repo ID
|-
| tokenizer || PreTrainedTokenizer || No || Tokenizer to save (recommended)
|-
| save_method || str || No || "merged_16bit" (default), "lora", or "merged_4bit"
|-
| push_to_hub || bool || No || Upload to HuggingFace Hub (default: False)
|-
| token || str || No || HuggingFace API token for Hub upload
|-
| max_shard_size || str || No || Max file size per shard (default: "5GB")
|-
| safe_serialization || bool || No || Use safetensors format (default: True)
|-
| maximum_memory_usage || float || No || GPU memory threshold (default: 0.9)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Files || Directory || Contains model.safetensors, config.json, tokenizer files
|-
| Return || Tuple[str, None] || (save_directory, None) on success
|}

== Usage Examples ==

=== Basic Merged Save ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# After training...
# Save merged model to 16-bit for GGUF conversion
model.save_pretrained_merged(
    "merged_model",          # Output directory
    tokenizer,               # Include tokenizer
    save_method="merged_16bit",
)

# Output structure:
# merged_model/
# ├── config.json
# ├── model.safetensors (or model-00001-of-00002.safetensors for large models)
# ├── tokenizer.json
# ├── tokenizer_config.json
# └── special_tokens_map.json
</syntaxhighlight>

=== Push to HuggingFace Hub ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Merge and upload to Hub
model.save_pretrained_merged(
    "your-username/my-merged-model",
    tokenizer,
    save_method="merged_16bit",
    push_to_hub=True,
    token="hf_...",
    private=False,
    commit_message="Fine-tuned with Unsloth",
)

# Model available at: https://huggingface.co/your-username/my-merged-model
</syntaxhighlight>

=== Save LoRA Adapters Only ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Save only LoRA adapters (fastest, smallest files)
model.save_pretrained_merged(
    "lora_adapter",
    tokenizer,
    save_method="lora",  # Only save adapter weights
)

# Output is ~50-200MB instead of multiple GB
# Can be loaded with PeftModel.from_pretrained()
</syntaxhighlight>

=== Memory-Efficient Saving for Large Models ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# For 70B+ models, use smaller shards and memory threshold
model.save_pretrained_merged(
    "large_merged_model",
    tokenizer,
    save_method="merged_16bit",
    max_shard_size="2GB",          # Smaller shards
    maximum_memory_usage=0.7,      # Leave headroom for merge
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:unslothai_unsloth_Weight_Merging]]

=== Requires Environment ===
* [[requires_env::Environment:unslothai_unsloth_CUDA]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:unslothai_unsloth_Memory_Optimization]]
