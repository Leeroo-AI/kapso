{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::NLP]], [[domain::Inference]], [[domain::LLM_Serving]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

Concrete tool for initializing a vLLM LLM instance with model loading, tokenizer setup, and engine configuration for offline batch inference.

=== Description ===

The `LLM` class is the primary entry point for offline inference in vLLM. It initializes an `LLMEngine` under the hood, handling model loading from HuggingFace Hub or local paths, tokenizer initialization, GPU memory management, and tensor parallelism configuration. This class is designed for batch processing scenarios where prompts are collected and processed together rather than streaming.

=== Usage ===

Use this class when you need to run batch inference on a large number of prompts offline. It is the recommended entry point for:
- Batch text generation jobs
- Model evaluation pipelines
- Offline data processing with LLMs
- Development and testing of prompts

NOT for online serving (use `AsyncLLMEngine` or `vllm serve` instead).

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project/vllm]
* '''File:''' vllm/entrypoints/llm.py
* '''Lines:''' L190-337

=== Signature ===
<syntaxhighlight lang="python">
class LLM:
    def __init__(
        self,
        model: str,
        *,
        runner: RunnerOption = "auto",
        tokenizer: str | None = None,
        tokenizer_mode: TokenizerMode | str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: ModelDType = "auto",
        quantization: QuantizationMethods | None = None,
        revision: str | None = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: float = 4,
        enforce_eager: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model: HuggingFace model name or path to local model.
            runner: Runner type ("auto", "generate", "pooling").
            tokenizer: Custom tokenizer path (defaults to model's tokenizer).
            tokenizer_mode: Tokenizer loading mode ("auto", "slow", "mistral").
            skip_tokenizer_init: Skip tokenizer initialization.
            trust_remote_code: Trust remote code in HuggingFace models.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            dtype: Model dtype ("auto", "float16", "bfloat16", "float32").
            quantization: Quantization method ("awq", "gptq", "fp8", etc.).
            revision: Model revision/branch from HuggingFace.
            seed: Random seed for reproducibility.
            gpu_memory_utilization: Fraction of GPU memory for KV cache (0.0-1.0).
            swap_space: CPU swap space in GB for KV cache offload.
            enforce_eager: Disable CUDA graph compilation.
            **kwargs: Additional arguments passed to EngineArgs.
        """
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
| model || str || Yes || HuggingFace model name (e.g., "meta-llama/Llama-2-7b-hf") or local path
|-
| tensor_parallel_size || int || No || Number of GPUs for tensor parallelism (default: 1)
|-
| dtype || str || No || Model precision ("auto", "float16", "bfloat16")
|-
| quantization || str || No || Quantization method ("awq", "gptq", "fp8", "squeezellm")
|-
| gpu_memory_utilization || float || No || GPU memory fraction for KV cache (default: 0.9)
|-
| trust_remote_code || bool || No || Allow custom model code from HuggingFace
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| LLM instance || LLM || Initialized LLM with loaded model, tokenizer, and engine
|-
| llm_engine || LLMEngine || Internal engine (accessible via `llm.llm_engine`)
|}

== Usage Examples ==

=== Basic Initialization ===
<syntaxhighlight lang="python">
from vllm import LLM

# Initialize with default settings
llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

# Access the tokenizer
tokenizer = llm.get_tokenizer()
</syntaxhighlight>

=== Multi-GPU with Quantization ===
<syntaxhighlight lang="python">
from vllm import LLM

# Initialize on 4 GPUs with AWQ quantization
llm = LLM(
    model="TheBloke/Llama-2-70B-AWQ",
    tensor_parallel_size=4,
    quantization="awq",
    gpu_memory_utilization=0.85,
    dtype="float16",
)
</syntaxhighlight>

=== Custom Memory Settings ===
<syntaxhighlight lang="python">
from vllm import LLM

# Conservative memory usage with swap
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    gpu_memory_utilization=0.7,  # Leave 30% GPU memory free
    swap_space=8,                 # 8GB CPU swap for KV cache
    enforce_eager=True,           # Disable CUDA graphs for debugging
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_LLM_Class_Initialization]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
