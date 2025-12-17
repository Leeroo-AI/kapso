# Implementation: unslothai_unsloth_save_pretrained_merged

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|SafeTensors|https://huggingface.co/docs/safetensors]]
|-
! Domains
| [[domain::NLP]], [[domain::Model_Export]], [[domain::Deployment]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Concrete tool for saving trained models by merging LoRA adapters with base weights and dequantizing to 16-bit precision.

=== Description ===

`model.save_pretrained_merged` performs three key operations:

1. **Dequantizes 4-bit weights** back to 16-bit precision
2. **Merges LoRA adapters** with base weights (W' = W + BA)
3. **Saves as safetensors** files compatible with standard HuggingFace loading

This produces a standalone model that can be loaded without Unsloth or LoRA dependencies, suitable for deployment or further conversion to GGUF.

=== Usage ===

Use this after training when you need:
- A merged model for deployment
- A base for GGUF conversion
- To share a model without LoRA dependencies

This is the primary export method for QLoRA-trained models.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/save.py
* '''Lines:''' L200-850

=== Signature ===
<syntaxhighlight lang="python">
def save_pretrained_merged(
    self,
    save_directory: Union[str, os.PathLike],
    tokenizer: Optional[PreTrainedTokenizer] = None,
    save_method: str = "merged_16bit",
    push_to_hub: bool = False,
    token: Optional[str] = None,
    private: Optional[bool] = None,
    max_shard_size: Union[int, str] = "5GB",
    safe_serialization: bool = True,
    **kwargs,
) -> Tuple[str, Optional[str]]:
    """
    Save model with LoRA merged and dequantized.

    Args:
        save_directory: Path or HuggingFace repo to save to
        tokenizer: Tokenizer to save alongside model
        save_method: "merged_16bit" or "merged_4bit" (not recommended)
        push_to_hub: Upload to HuggingFace Hub
        token: HuggingFace API token
        private: Make repo private on Hub
        max_shard_size: Maximum size per shard file

    Returns:
        Tuple of (save_directory, username if pushed)
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# save_pretrained_merged is added to model by Unsloth
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(...)
# model now has .save_pretrained_merged() method
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| save_directory || str || Yes || Local path or HuggingFace repo ID
|-
| tokenizer || PreTrainedTokenizer || No || Tokenizer to save with model
|-
| save_method || str || No (default: "merged_16bit") || Export precision
|-
| push_to_hub || bool || No (default: False) || Upload to HuggingFace Hub
|-
| token || str || No || HuggingFace API token (required if push_to_hub)
|-
| max_shard_size || str || No (default: "5GB") || Maximum file size per shard
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| save_directory || str || Path where model was saved
|-
| username || Optional[str] || HuggingFace username (if pushed to Hub)
|-
| Saved files || files || config.json, model.safetensors, tokenizer files
|}

== Usage Examples ==

=== Save Locally ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# After training...

# Save merged 16-bit model locally
model.save_pretrained_merged(
    "my_trained_model",
    tokenizer = tokenizer,
    save_method = "merged_16bit",
)

# Files created:
# my_trained_model/
# ├── config.json
# ├── model.safetensors
# ├── tokenizer.json
# ├── tokenizer_config.json
# └── special_tokens_map.json
</syntaxhighlight>

=== Push to HuggingFace Hub ===
<syntaxhighlight lang="python">
# Save and push to Hub
model.save_pretrained_merged(
    "username/my-llama-finetuned",  # Hub repo ID
    tokenizer = tokenizer,
    save_method = "merged_16bit",
    push_to_hub = True,
    token = "hf_your_token",
    private = False,  # Public repo
)

# Model now available at https://huggingface.co/username/my-llama-finetuned
</syntaxhighlight>

=== Save for GGUF Conversion ===
<syntaxhighlight lang="python">
# Save 16-bit merged model as prerequisite for GGUF
model.save_pretrained_merged(
    "model_for_gguf",
    tokenizer = tokenizer,
    save_method = "merged_16bit",
)

# Now can convert to GGUF
model.save_pretrained_gguf(
    "model_for_gguf",
    tokenizer = tokenizer,
    quantization_method = "q4_k_m",
)
</syntaxhighlight>

=== Large Model Sharding ===
<syntaxhighlight lang="python">
# For large models, control shard size
model.save_pretrained_merged(
    "large_model",
    tokenizer = tokenizer,
    save_method = "merged_16bit",
    max_shard_size = "2GB",  # Smaller shards for limited disk
)

# Creates multiple files:
# model-00001-of-00003.safetensors
# model-00002-of-00003.safetensors
# model-00003-of-00003.safetensors
</syntaxhighlight>

=== Memory-Constrained Export ===
<syntaxhighlight lang="python">
import torch
import gc

# Clear memory before saving (important for large models)
torch.cuda.empty_cache()
gc.collect()

# Save with memory optimization
model.save_pretrained_merged(
    "output_model",
    tokenizer = tokenizer,
    save_method = "merged_16bit",
)

# Clean up after saving
torch.cuda.empty_cache()
gc.collect()
</syntaxhighlight>

== Save Methods Comparison ==

| Save Method | Output Format | Size | Use Case |
|-------------|---------------|------|----------|
| `"merged_16bit"` | float16 safetensors | 2x params bytes | Standard export, GGUF base |
| `"merged_4bit"` | 4-bit safetensors | 0.5x params bytes | Space-constrained (quality loss!) |
| `"lora"` | LoRA adapters only | ~1% of model | Lightweight sharing |

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:unslothai_unsloth_Model_Saving]]
* [[implements::Principle:unslothai_unsloth_Merged_Export]]

=== Requires Environment ===
* [[requires_env::Environment:unslothai_unsloth_CUDA]]
