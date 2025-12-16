{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Docs|https://docs.unsloth.ai]]
* [[source::Doc|GGUF Guide|https://docs.unsloth.ai/basics/running-and-saving-models/saving-to-gguf]]
* [[source::Repo|llama.cpp|https://github.com/ggml-org/llama.cpp]]
|-
! Domains
| [[domain::LLMs]], [[domain::GGUF]], [[domain::Quantization]], [[domain::Deployment]]
|-
! Last Updated
| [[last_updated::2025-12-16 14:00 GMT]]
|}

== Overview ==
Concrete tool for converting fine-tuned models to GGUF format for llama.cpp and Ollama deployment provided by the Unsloth library.

=== Description ===
`model.save_pretrained_gguf()` is a method that handles the complete GGUF conversion pipeline:

1. **LoRA merging**: Automatically merges adapter weights into base model
2. **llama.cpp installation**: Downloads and compiles llama.cpp if needed
3. **Format conversion**: Converts HuggingFace format to GGUF using llama.cpp scripts
4. **Quantization**: Applies various quantization methods (q4_k_m, q8_0, f16, etc.)
5. **Tokenizer fixing**: Repairs sentencepiece tokenizers for GGUF compatibility
6. **Ollama Modelfile**: Generates appropriate Modelfile for Ollama deployment

The method supports 20+ quantization methods with different quality/size tradeoffs, from full precision (f16/bf16) to aggressive compression (q2_k).

=== Usage ===
Use this method when you need to:
- Deploy models with llama.cpp for CPU/consumer hardware inference
- Create Ollama-compatible model files
- Reduce model size with quantization
- Export for edge deployment

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unslothai_unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/save.py#L1776-L2000 unsloth/save.py]
* '''Lines:''' 1776-2000 (main function), 1061-1200 (save_to_gguf helper)

Source Files: unsloth/save.py:L1776-L2000; unsloth/save.py:L1061-L1200

=== Signature ===
<syntaxhighlight lang="python">
def save_pretrained_gguf(
    self,
    save_directory: Union[str, os.PathLike],
    tokenizer: Optional[PreTrainedTokenizer] = None,
    quantization_method: Union[str, List[str]] = "fast_quantized",
    first_conversion: Optional[str] = None,
    push_to_hub: bool = False,
    token: Optional[Union[str, bool]] = None,
    private: Optional[bool] = None,
    is_main_process: bool = True,
    state_dict: Optional[dict] = None,
    save_function: Callable = torch.save,
    max_shard_size: Union[int, str] = "5GB",
    safe_serialization: bool = True,
    variant: Optional[str] = None,
    save_peft_format: bool = True,
    tags: Optional[List[str]] = None,
    temporary_location: str = "_unsloth_temporary_saved_buffers",
    maximum_memory_usage: float = 0.85,
) -> Dict[str, Any]:
    """
    Convert model to GGUF format for llama.cpp/Ollama deployment.

    Args:
        save_directory: Output directory for GGUF files
        tokenizer: Tokenizer (required for GGUF conversion)
        quantization_method: Quantization type(s) - see table below
        first_conversion: Initial conversion format (f16/bf16)
        push_to_hub: Use push_to_hub_gguf() instead
        max_shard_size: Temporary shard size during merging
        maximum_memory_usage: RAM fraction for conversion (0.0-0.95)

    Returns:
        Dict with gguf_files, modelfile_location, save_directory
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
# Method is automatically attached to model after loading
# model.save_pretrained_gguf(...)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| save_directory || str || Yes || Output directory for GGUF files
|-
| tokenizer || PreTrainedTokenizer || Yes || Tokenizer for conversion
|-
| quantization_method || str or List[str] || No (default: "fast_quantized") || Quantization method(s)
|-
| first_conversion || str || No (auto) || Initial format (f16 or bf16)
|-
| maximum_memory_usage || float || No (default: 0.85) || RAM fraction limit
|}

=== Quantization Methods ===
{| class="wikitable"
|-
! Method !! Description !! Use Case
|-
| not_quantized / f16 || Float16, 100% accuracy || Development, accuracy testing
|-
| bf16 || BFloat16, 100% accuracy || Ampere+ GPUs
|-
| fast_quantized / q8_0 || 8-bit quantization || High quality, moderate size
|-
| quantized / q4_k_m || 4-bit with Q6_K for important layers || '''Recommended''' balance
|-
| q5_k_m || 5-bit with Q6_K for important layers || Higher quality than q4_k_m
|-
| q4_k_s || 4-bit for all tensors || Smaller file size
|-
| q3_k_m || 3-bit mixed quantization || Aggressive compression
|-
| q2_k || 2-bit quantization || Extreme compression
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| gguf_files || List[str] || Paths to generated .gguf files
|-
| modelfile_location || str || Path to generated Ollama Modelfile
|-
| save_directory || str || Output directory used
|-
| .gguf files || Files || Quantized model files
|-
| Modelfile || File || Ollama deployment template
|}

== Usage Examples ==

=== Basic GGUF Export ===
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

# Export to GGUF with recommended quantization
model.save_pretrained_gguf(
    "model_gguf",
    tokenizer,
    quantization_method="q4_k_m",  # Good balance of size/quality
)

# Result: model_gguf/model-q4_k_m.gguf + Modelfile
</syntaxhighlight>

=== Multiple Quantization Methods ===
<syntaxhighlight lang="python">
# Generate multiple quantized versions at once
model.save_pretrained_gguf(
    "model_gguf",
    tokenizer,
    quantization_method=["q4_k_m", "q5_k_m", "q8_0"],  # List of methods
)

# Result: Creates three .gguf files
# - model_gguf/model-q4_k_m.gguf (smallest)
# - model_gguf/model-q5_k_m.gguf (medium)
# - model_gguf/model-q8_0.gguf (highest quality)
</syntaxhighlight>

=== Full Precision Export ===
<syntaxhighlight lang="python">
# Export without quantization for maximum accuracy
model.save_pretrained_gguf(
    "model_gguf_f16",
    tokenizer,
    quantization_method="f16",  # or "bf16" for bfloat16
)

# Result: Large file but 100% accuracy retained
</syntaxhighlight>

=== Deploy with Ollama ===
<syntaxhighlight lang="python">
# Export and then deploy locally with Ollama
model.save_pretrained_gguf(
    "model_gguf",
    tokenizer,
    quantization_method="q4_k_m",
)

# The Modelfile is generated automatically
# Deploy with:
# $ cd model_gguf
# $ ollama create my-model -f Modelfile
# $ ollama run my-model
</syntaxhighlight>

=== Push GGUF to Hub ===
<syntaxhighlight lang="python">
# Use push_to_hub_gguf for uploading
model.push_to_hub_gguf(
    "your-username/model-GGUF",
    tokenizer,
    quantization_method="q4_k_m",
    token="hf_...",
)

# Result: GGUF file uploaded to HuggingFace Hub
</syntaxhighlight>

=== Memory-Constrained Conversion ===
<syntaxhighlight lang="python">
# For systems with limited RAM
model.save_pretrained_gguf(
    "model_gguf",
    tokenizer,
    quantization_method="q4_k_m",
    maximum_memory_usage=0.6,  # Use only 60% of RAM
    max_shard_size="2GB",  # Smaller intermediate shards
)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:unslothai_unsloth_llama_cpp]]
* [[uses_heuristic::Heuristic:unslothai_unsloth_Quantization_Method_Selection]]
