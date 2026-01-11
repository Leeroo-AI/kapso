# Implementation: unsloth_save_model_merged

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Model_Serialization]], [[domain::GGUF]], [[domain::LoRA]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Concrete tool for merging LoRA adapters into base model weights in preparation for GGUF export.

=== Description ===

`save_pretrained_merged` merges LoRA adapters into the base model and saves to float16/bfloat16 format. For GGUF export, this creates the intermediate representation that llama.cpp converts to GGUF.

=== Usage ===

Call before GGUF export to create merged 16-bit weights from a 4-bit LoRA model.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/save.py
* '''Lines:''' L235-860

=== Signature ===
<syntaxhighlight lang="python">
def save_pretrained_merged(
    self,
    save_directory: str,
    tokenizer = None,
    save_method: str = "merged_16bit",  # For GGUF prep
    push_to_hub: bool = False,
    token: Optional[str] = None,
    is_main_process: bool = True,
    state_dict: Optional[dict] = None,
    save_function = safetensors.torch.save_file,
    max_shard_size: Union[int, str] = "5GB",
    safe_serialization: bool = True,
    variant: Optional[str] = None,
    save_peft_format: bool = True,
    tags: Optional[List[str]] = None,
    temporary_location: str = "_unsloth_temporary_saved_buffers",
    maximum_memory_usage: float = 0.85,
) -> Tuple[str, Optional[str]]:
    """
    Merge LoRA adapters and save model.

    For GGUF export, use save_method="merged_16bit" to create
    float16 weights that llama.cpp can convert.
    """
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Parameter !! Type !! Description
|-
| save_directory || str || Output directory for merged model
|-
| tokenizer || PreTrainedTokenizer || Tokenizer to save alongside model
|-
| save_method || str || "merged_16bit" for GGUF prep
|-
| maximum_memory_usage || float || RAM usage limit (0.85 = 85%)
|}

=== Outputs ===
{| class="wikitable"
|-
! Output !! Type !! Description
|-
| save_directory || str || Path where model was saved
|-
| commit_url || Optional[str] || Hub URL if push_to_hub=True
|}

== Usage Examples ==

=== Merge for GGUF Export ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# After training...
# Merge LoRA and save as 16-bit (required for GGUF)
model.save_pretrained_merged(
    "./merged_model",
    tokenizer,
    save_method = "merged_16bit",
    maximum_memory_usage = 0.9,  # Use up to 90% RAM
)
# Now ready for: model.save_pretrained_gguf(...)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_Model_Preparation]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]

