# Implementation: push_to_hub_gguf

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Model_Sharing]], [[domain::HuggingFace_Hub]], [[domain::GGUF]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Concrete tool for uploading GGUF models to HuggingFace Hub.

=== Description ===

`push_to_hub_gguf` converts models to GGUF format and uploads directly to HuggingFace Hub. It combines:
* GGUF conversion (via save_pretrained_gguf)
* Repository creation
* File upload with progress tracking
* README generation

=== Usage ===

Call after training to export and share GGUF models in one step.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/save.py
* '''Lines:''' L2060-2300

=== Signature ===
<syntaxhighlight lang="python">
def push_to_hub_gguf(
    self,
    repo_id: str,
    tokenizer = None,
    quantization_method: Union[str, List[str]] = "fast_quantized",
    first_conversion: Optional[str] = None,
    use_temp_dir: Optional[bool] = None,
    commit_message: Optional[str] = "Trained with Unsloth",
    private: Optional[bool] = None,
    token: Union[bool, str, None] = None,
    max_shard_size: Union[int, str] = "5GB",
    create_pr: bool = False,
    safe_serialization: bool = True,
    revision: str = None,
    commit_description: str = "Upload model trained with Unsloth 2x faster",
    tags: Optional[List[str]] = None,
    temporary_location: str = "_unsloth_temporary_saved_buffers",
    maximum_memory_usage: float = 0.85,
) -> str:
    """
    Convert to GGUF and upload to HuggingFace Hub.

    Returns:
        URL of the uploaded model repository
    """
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Parameter !! Type !! Description
|-
| repo_id || str || HuggingFace repo ID (user/model-name)
|-
| tokenizer || PreTrainedTokenizer || Model tokenizer
|-
| quantization_method || Union[str, List[str]] || Quant method(s) to upload
|-
| private || Optional[bool] || Create private repository
|-
| token || Union[bool, str, None] || HuggingFace API token
|}

=== Outputs ===
{| class="wikitable"
|-
! Output !! Type !! Description
|-
| repo_url || str || URL of uploaded repository
|}

== Usage Examples ==

=== Basic Hub Upload ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# After training...
# Upload to HuggingFace Hub with q4_k_m
model.push_to_hub_gguf(
    "username/my-model-GGUF",
    tokenizer,
    quantization_method = "q4_k_m",
    token = "hf_...",  # Your HF token
)
</syntaxhighlight>

=== Multiple Quantizations ===
<syntaxhighlight lang="python">
# Upload multiple quantization levels
model.push_to_hub_gguf(
    "username/my-model-GGUF",
    tokenizer,
    quantization_method = ["q4_k_m", "q8_0", "q5_k_m"],
    private = False,
    commit_message = "Add GGUF quantizations",
)
</syntaxhighlight>

=== Private Repository ===
<syntaxhighlight lang="python">
# Create private GGUF repository
model.push_to_hub_gguf(
    "username/my-private-model-GGUF",
    tokenizer,
    quantization_method = "q4_k_m",
    private = True,
    token = "hf_...",
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_GGUF_Hub_Upload]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:Unslothai_Unsloth_GGUF_Quantization_Selection_Tip]]

