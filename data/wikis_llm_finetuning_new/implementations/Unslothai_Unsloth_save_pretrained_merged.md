# Implementation: save_pretrained_merged

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
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

Concrete tool for saving trained models with optional LoRA merging provided by the Unsloth library.

=== Description ===

`model.save_pretrained_merged` saves a trained model with options to:

* Save LoRA adapters only (smallest, requires base model)
* Merge LoRA into 16-bit weights (for GGUF conversion)
* Merge LoRA into 4-bit weights (for memory-efficient inference)

The function handles the complex process of dequantizing 4-bit weights, merging LoRA deltas, and serializing the result with safetensors format.

=== Usage ===

Call this method on your trained model after training is complete. Choose the save_method based on your deployment target. Use "lora" for sharing adapters, "merged_16bit" for GGUF conversion, or "merged_4bit" for bitsandbytes inference.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/save.py
* '''Lines:''' 1337-1420 (unsloth_save_pretrained_merged), 234-858 (unsloth_save_model core logic)

=== Signature ===
<syntaxhighlight lang="python">
def save_pretrained_merged(
    self,
    save_directory: Union[str, os.PathLike],
    tokenizer: Optional[PreTrainedTokenizer] = None,
    save_method: str = "merged_16bit",
    push_to_hub: bool = False,
    token: Optional[Union[str, bool]] = None,
    is_main_process: bool = True,
    state_dict: Optional[dict] = None,
    save_function: Callable = torch.save,
    max_shard_size: Union[int, str] = "5GB",
    safe_serialization: bool = True,
    variant: Optional[str] = None,
    save_peft_format: bool = True,
    tags: List[str] = None,
    temporary_location: str = "_unsloth_temporary_saved_buffers",
    maximum_memory_usage: float = 0.75,
) -> None:
    """
    Save the model with optional LoRA merging.

    Args:
        save_directory: Output directory or HuggingFace repo ID
        tokenizer: Tokenizer to save alongside model
        save_method: One of "lora", "merged_16bit", "merged_4bit"
        push_to_hub: Upload to HuggingFace Hub
        token: HuggingFace API token
        max_shard_size: Maximum size per shard file
        safe_serialization: Use safetensors format
        maximum_memory_usage: Max GPU memory fraction for merging

    Save Methods:
        - "lora": Save adapter weights only (~10-100MB)
        - "merged_16bit": Merge into float16 (~2GB/B params)
        - "merged_4bit": Merge into int4 (~0.5GB/B params)
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Method is attached to model after loading
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(...)
model = FastLanguageModel.get_peft_model(model, ...)
# ... training ...
model.save_pretrained_merged(save_directory, tokenizer, save_method="merged_16bit")
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| save_directory || str || Yes || Output directory path or HuggingFace repo ID
|-
| tokenizer || PreTrainedTokenizer || No || Tokenizer to save (recommended)
|-
| save_method || str || No || "lora", "merged_16bit", or "merged_4bit" (default: "merged_16bit")
|-
| push_to_hub || bool || No || Upload to HuggingFace Hub (default: False)
|-
| token || str || No || HuggingFace API token for private repos
|-
| max_shard_size || str || No || Max file size per shard (default: "5GB")
|-
| safe_serialization || bool || No || Use safetensors format (default: True)
|-
| maximum_memory_usage || float || No || Max GPU memory for merging (default: 0.75)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| (files) || Various || Model files saved to save_directory
|-
| config.json || JSON || Model configuration
|-
| model.safetensors || Binary || Model weights (or sharded files)
|-
| tokenizer files || Various || Tokenizer configuration and vocabulary
|-
| adapter_config.json || JSON || (lora only) LoRA configuration
|-
| adapter_model.safetensors || Binary || (lora only) LoRA weights
|}

== Usage Examples ==

=== Save LoRA Adapters Only ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=16)

# ... training ...

# Save only the LoRA adapters (~50MB)
model.save_pretrained_merged(
    save_directory="./lora_adapters",
    tokenizer=tokenizer,
    save_method="lora",
)

# Directory contains:
# - adapter_config.json
# - adapter_model.safetensors
</syntaxhighlight>

=== Save Merged 16-bit Model ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=16)

# ... training ...

# Merge and save as 16-bit (for GGUF conversion)
model.save_pretrained_merged(
    save_directory="./merged_model",
    tokenizer=tokenizer,
    save_method="merged_16bit",
)

# Directory contains:
# - config.json
# - model.safetensors (or sharded files)
# - tokenizer files
</syntaxhighlight>

=== Push to HuggingFace Hub ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=16)

# ... training ...

# Push merged model to HuggingFace Hub
model.save_pretrained_merged(
    save_directory="username/my-finetuned-model",
    tokenizer=tokenizer,
    save_method="merged_16bit",
    push_to_hub=True,
    token="hf_xxx",
)
</syntaxhighlight>

=== Using push_to_hub_merged Shortcut ===
<syntaxhighlight lang="python">
# Alternative: use the push_to_hub_merged method directly
model.push_to_hub_merged(
    repo_id="username/my-model",
    tokenizer=tokenizer,
    save_method="merged_16bit",
    token="hf_xxx",
    private=True,
)
</syntaxhighlight>

=== Save 4-bit Merged Model ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=16)

# ... training ...

# Merge and re-quantize to 4-bit (requires bitsandbytes at inference)
model.save_pretrained_merged(
    save_directory="./merged_4bit",
    tokenizer=tokenizer,
    save_method="merged_4bit",
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_Model_Saving]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_11]]
