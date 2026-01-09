# Implementation: save_to_gguf

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Model_Export]], [[domain::GGUF]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Concrete tool for converting fine-tuned models to GGUF format using llama.cpp.

=== Description ===

`save_pretrained_gguf` orchestrates the full GGUF conversion pipeline:
1. Merges LoRA adapters to float16
2. Installs llama.cpp if needed
3. Converts HF format to GGUF
4. Applies quantization
5. Creates Ollama Modelfile

=== Usage ===

Call after training to export model for llama.cpp/Ollama deployment.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/save.py
* '''Lines:''' L1070-1300 (save_to_gguf), L1760-2058 (save_pretrained_gguf)

=== Signature ===
<syntaxhighlight lang="python">
def save_pretrained_gguf(
    self,
    save_directory: str,
    tokenizer = None,
    quantization_method: Union[str, List[str]] = "fast_quantized",
    first_conversion: Optional[str] = None,
    push_to_hub: bool = False,
    token: Optional[str] = None,
    is_main_process: bool = True,
    state_dict: Optional[dict] = None,
    save_function = safetensors.torch.save_file,
    max_shard_size: Union[int, str] = "5GB",
    safe_serialization: bool = True,
    variant: Optional[str] = None,
    save_peft_format: bool = True,
    tags: List[str] = None,
    temporary_location: str = "_unsloth_temporary_saved_buffers",
    maximum_memory_usage: float = 0.85,
) -> dict:
    """
    Convert model to GGUF format.

    Returns dict with:
    - save_directory: Output path
    - gguf_files: List of GGUF file paths
    - modelfile_location: Ollama Modelfile path
    - is_vlm: Whether model is vision-language
    """
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Parameter !! Type !! Description
|-
| save_directory || str || Output directory for GGUF files
|-
| tokenizer || PreTrainedTokenizer || Model tokenizer
|-
| quantization_method || Union[str, List[str]] || Quant method(s) from ALLOWED_QUANTS
|-
| first_conversion || Optional[str] || Initial precision (auto-detected)
|}

=== Outputs ===
{| class="wikitable"
|-
! Output !! Type !! Description
|-
| gguf_files || List[str] || Paths to generated GGUF files
|-
| modelfile_location || str || Path to Ollama Modelfile
|-
| is_vlm || bool || Whether model is VLM
|}

== Usage Examples ==

=== Basic GGUF Export ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# After training...
# Export to GGUF with q4_k_m quantization
model.save_pretrained_gguf(
    "./model_gguf",
    tokenizer,
    quantization_method = "q4_k_m",
)
# Creates: ./model_gguf/model-Q4_K_M.gguf
</syntaxhighlight>

=== Multiple Quantizations ===
<syntaxhighlight lang="python">
# Export multiple quantization levels at once
model.save_pretrained_gguf(
    "./model_gguf",
    tokenizer,
    quantization_method = ["q4_k_m", "q8_0", "q5_k_m"],
)
# Creates: model-Q4_K_M.gguf, model-Q8_0.gguf, model-Q5_K_M.gguf
</syntaxhighlight>

=== VLM Export (Partial Support) ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel

# After training VLM...
# Note: GGUF VLM support is limited to LLaVA-style architectures
model.save_pretrained_gguf(
    "./vlm_gguf",
    processor,
    quantization_method = "q4_k_m",
)
# Creates: model-Q4_K_M.gguf + mmproj file for vision encoder
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_GGUF_Export]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:Unslothai_Unsloth_GGUF_Quantization_Selection_Tip]]

