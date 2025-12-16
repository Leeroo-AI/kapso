{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Docs|https://docs.unsloth.ai]]
* [[source::Doc|Saving Guide|https://docs.unsloth.ai/basics/running-and-saving-models]]
|-
! Domains
| [[domain::LLMs]], [[domain::Model_Export]], [[domain::LoRA]], [[domain::Deployment]]
|-
! Last Updated
| [[last_updated::2025-12-16 14:00 GMT]]
|}

== Overview ==
Concrete tool for merging LoRA adapters into base model weights and saving to HuggingFace format provided by the Unsloth library.

=== Description ===
`model.save_pretrained_merged()` is a method dynamically attached to models loaded via Unsloth that handles the complete process of:

1. **LoRA merging**: Fuses adapter weights (A, B matrices) back into base weights using `W_merged = W_base + (A @ B) * scaling`
2. **Dequantization**: Converts 4-bit quantized weights to float16/bfloat16 for saving
3. **Memory-efficient saving**: Processes weights layer-by-layer to minimize RAM usage
4. **HuggingFace format**: Saves in standard safetensors format compatible with transformers
5. **Hub upload**: Optional direct push to HuggingFace Hub

The method supports three save modes:
- `"merged_16bit"`: Merge LoRA and save as float16 (recommended for GGUF conversion)
- `"merged_4bit"`: Merge LoRA and save as 4-bit (useful for continued training/DPO)
- `"lora"`: Save only the LoRA adapter weights (smallest files)

=== Usage ===
Use this method when you need to:
- Export a fine-tuned model for deployment
- Prepare a model for GGUF/llama.cpp conversion
- Share a model on HuggingFace Hub
- Create a checkpoint for continued training

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unslothai_unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/save.py#L2653-L2693 unsloth/save.py]
* '''Lines:''' 2653-2693 (wrapper), 228-506 (core implementation)

Source Files: unsloth/save.py:L228-L506; unsloth/save.py:L2653-L2693

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
    tags: Optional[List[str]] = None,
    temporary_location: str = "_unsloth_temporary_saved_buffers",
    maximum_memory_usage: float = 0.75,
) -> Tuple[str, Optional[str]]:
    """
    Merge LoRA adapters and save model to HuggingFace format.

    Args:
        save_directory: Local path or HuggingFace repo ID
        tokenizer: Tokenizer to save alongside model
        save_method: "merged_16bit", "merged_4bit", or "lora"
        push_to_hub: Upload to HuggingFace Hub
        token: HuggingFace API token
        max_shard_size: Maximum size per model shard
        safe_serialization: Use safetensors format
        tags: Custom tags for the model
        maximum_memory_usage: Max RAM fraction for merging (0.0-0.95)

    Returns:
        Tuple of (save_directory, None)
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
# Method is automatically attached to model after loading
# model.save_pretrained_merged(...)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| save_directory || str || Yes || Local path or HuggingFace repo ID
|-
| tokenizer || PreTrainedTokenizer || No || Tokenizer to save (recommended)
|-
| save_method || str || No (default: "merged_16bit") || How to save: "merged_16bit", "merged_4bit", "lora"
|-
| push_to_hub || bool || No (default: False) || Upload to HuggingFace Hub
|-
| token || str || No || HuggingFace API token (required if push_to_hub=True)
|-
| max_shard_size || str || No (default: "5GB") || Maximum shard file size
|-
| safe_serialization || bool || No (default: True) || Use safetensors format
|-
| maximum_memory_usage || float || No (default: 0.75) || RAM fraction limit
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| save_directory || str || Path where model was saved
|-
| Files created || safetensors/bin || Model weights in sharded format
|-
| config.json || JSON || Model configuration
|-
| tokenizer files || Various || Tokenizer vocabulary and config
|}

== Usage Examples ==

=== Save Merged 16-bit Model ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load and fine-tune model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=16)

# ... training code ...

# Save merged model (recommended for deployment)
model.save_pretrained_merged(
    "model_merged_16bit",
    tokenizer,
    save_method="merged_16bit",
)

# Result: Full model in safetensors format ready for vLLM/SGLang
</syntaxhighlight>

=== Save LoRA Adapters Only ===
<syntaxhighlight lang="python">
# Save just the LoRA weights (smallest file size)
model.save_pretrained_merged(
    "model_lora_only",
    tokenizer,
    save_method="lora",
)

# Result: Only adapter_config.json and adapter_model.safetensors
# Base model weights are not included - much smaller!
</syntaxhighlight>

=== Push to HuggingFace Hub ===
<syntaxhighlight lang="python">
# Save and upload to HuggingFace
model.save_pretrained_merged(
    "your-username/my-finetuned-llama",
    tokenizer,
    save_method="merged_16bit",
    push_to_hub=True,
    token="hf_...",  # Your HuggingFace token
)

# Model available at https://huggingface.co/your-username/my-finetuned-llama
</syntaxhighlight>

=== Save for DPO/Continued Training ===
<syntaxhighlight lang="python">
# Save as 4-bit for DPO or continued training
# Note: Use merged_4bit_forced to confirm
model.save_pretrained_merged(
    "model_merged_4bit",
    tokenizer,
    save_method="merged_4bit_forced",  # Must use _forced suffix
)

# Result: 4-bit quantized model for efficient inference or DPO training
</syntaxhighlight>

=== Memory-Constrained Saving ===
<syntaxhighlight lang="python">
# For systems with limited RAM, reduce memory usage
model.save_pretrained_merged(
    "model_output",
    tokenizer,
    save_method="merged_16bit",
    max_shard_size="2GB",  # Smaller shards
    maximum_memory_usage=0.5,  # Use less RAM (50%)
)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:unslothai_unsloth_Storage]]
* [[uses_heuristic::Heuristic:unslothai_unsloth_Memory_Management]]
