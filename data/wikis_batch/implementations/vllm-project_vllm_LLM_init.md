# LLM.__init__

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::Inference]], [[domain::Model_Loading]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:30 GMT]]
|}

== Overview ==

Concrete tool for instantiating the vLLM offline inference engine with model weights, tokenizer, and configured memory allocation.

=== Description ===

The `LLM` class constructor initializes the complete inference pipeline including:
* Loading model weights from HuggingFace or local path
* Initializing the tokenizer
* Allocating KV cache memory using PagedAttention
* Setting up worker processes for distributed execution

This is the primary entry point for offline batch inference in vLLM.

=== Usage ===

Use `LLM` constructor for offline batch inference scenarios where you want to:
* Process multiple prompts with automatic batching
* Maximize throughput without real-time latency requirements
* Use a simple synchronous API

For online serving with streaming, use `vllm serve` or `AsyncLLMEngine` instead.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm]
* '''File:''' vllm/entrypoints/llm.py

=== Signature ===
<syntaxhighlight lang="python">
class LLM:
    """An LLM for generating texts from given prompts and sampling parameters."""

    def __init__(
        self,
        model: str,
        *,
        runner: RunnerOption = "auto",
        convert: ConvertOption = "auto",
        tokenizer: str | None = None,
        tokenizer_mode: TokenizerMode | str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        allowed_local_media_path: str = "",
        allowed_media_domains: list[str] | None = None,
        tensor_parallel_size: int = 1,
        dtype: ModelDType = "auto",
        quantization: QuantizationMethods | None = None,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: float = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: bool = False,
        disable_custom_all_reduce: bool = False,
        hf_token: bool | str | None = None,
        hf_overrides: HfOverrides | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """LLM constructor."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm import LLM
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || str || Yes || HuggingFace model name or local path
|-
| tokenizer || str || No || Tokenizer name/path (defaults to model)
|-
| tensor_parallel_size || int || No || GPUs for tensor parallelism (default: 1)
|-
| gpu_memory_utilization || float || No || GPU memory fraction (default: 0.9)
|-
| dtype || str || No || Weight dtype: "auto", "float16", "bfloat16"
|-
| quantization || str || No || Quantization: "awq", "gptq", "fp8", etc.
|-
| trust_remote_code || bool || No || Trust HuggingFace remote code (default: False)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| LLM || LLM || Initialized inference engine instance
|-
| llm_engine || LLMEngine || Internal engine (accessible via attribute)
|}

== Usage Examples ==

=== Basic Model Loading ===
<syntaxhighlight lang="python">
from vllm import LLM

# Load model with default settings
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

# Model is ready for inference
outputs = llm.generate(["Hello, world!"])
</syntaxhighlight>

=== Multi-GPU Loading ===
<syntaxhighlight lang="python">
from vllm import LLM

# Load 70B model across 4 GPUs
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.85,
)
</syntaxhighlight>

=== Quantized Model Loading ===
<syntaxhighlight lang="python">
from vllm import LLM

# Load AWQ quantized model
llm = LLM(
    model="TheBloke/Llama-2-70B-AWQ",
    quantization="awq",
    dtype="float16",
)
</syntaxhighlight>

=== Memory-Constrained Loading ===
<syntaxhighlight lang="python">
from vllm import LLM

# Limit GPU memory usage for smaller KV cache
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    gpu_memory_utilization=0.5,  # Only use 50% GPU memory
    max_model_len=2048,          # Limit context length
)
</syntaxhighlight>

=== Loading with Custom Tokenizer ===
<syntaxhighlight lang="python">
from vllm import LLM

# Use separate tokenizer
llm = LLM(
    model="my-org/custom-model",
    tokenizer="meta-llama/Llama-3.1-8B-Instruct",
    trust_remote_code=True,
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_Model_Loading]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_CUDA_Environment]]
