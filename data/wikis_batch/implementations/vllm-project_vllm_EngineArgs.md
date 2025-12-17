# EngineArgs

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::Inference]], [[domain::Configuration]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:30 GMT]]
|}

== Overview ==

Concrete tool for configuring vLLM engine parameters including model path, memory settings, and parallelism options.

=== Description ===

`EngineArgs` is a dataclass that encapsulates all configuration options for the vLLM inference engine. It aggregates settings for model loading, memory management, parallelism, quantization, and various optimization features into a single configuration object.

This class serves as the single source of truth for engine configuration and is used internally by both the `LLM` class (offline inference) and the API server (online serving).

=== Usage ===

Use `EngineArgs` when you need programmatic access to engine configuration, particularly in advanced scenarios like custom engine initialization or configuration introspection. For standard offline inference, prefer using `LLM` constructor parameters directly.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm]
* '''File:''' vllm/engine/arg_utils.py

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class EngineArgs:
    """Arguments for vLLM engine."""
    model: str
    tokenizer: str | None = None
    tokenizer_mode: TokenizerMode | str = "auto"
    skip_tokenizer_init: bool = False
    trust_remote_code: bool = False
    tensor_parallel_size: int = 1
    dtype: ModelDType = "auto"
    quantization: QuantizationMethods | None = None
    revision: str | None = None
    tokenizer_revision: str | None = None
    seed: int = 0
    gpu_memory_utilization: float = 0.9
    swap_space: float = 4
    cpu_offload_gb: float = 0
    enforce_eager: bool = False
    max_model_len: int | None = None
    # ... many more parameters
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm.engine.arg_utils import EngineArgs
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || str || Yes || HuggingFace model name or local path
|-
| tensor_parallel_size || int || No || Number of GPUs for tensor parallelism (default: 1)
|-
| gpu_memory_utilization || float || No || Fraction of GPU memory to use (default: 0.9)
|-
| dtype || str || No || Data type for weights: "auto", "float16", "bfloat16", "float32"
|-
| quantization || str || No || Quantization method: "awq", "gptq", "fp8", etc.
|-
| max_model_len || int || No || Maximum sequence length (derived from model config if None)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| EngineArgs || dataclass || Configuration object for engine initialization
|-
| create_engine_config() || VllmConfig || Converts to full engine configuration
|}

== Usage Examples ==

=== Basic Configuration ===
<syntaxhighlight lang="python">
from vllm.engine.arg_utils import EngineArgs

# Configure for single-GPU inference
engine_args = EngineArgs(
    model="meta-llama/Llama-3.1-8B-Instruct",
    gpu_memory_utilization=0.9,
    max_model_len=4096,
)
</syntaxhighlight>

=== Multi-GPU Configuration ===
<syntaxhighlight lang="python">
from vllm.engine.arg_utils import EngineArgs

# Configure for 4-GPU tensor parallelism
engine_args = EngineArgs(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.85,
    dtype="bfloat16",
)
</syntaxhighlight>

=== Quantized Model Configuration ===
<syntaxhighlight lang="python">
from vllm.engine.arg_utils import EngineArgs

# Configure for AWQ quantized model
engine_args = EngineArgs(
    model="TheBloke/Llama-2-70B-AWQ",
    quantization="awq",
    gpu_memory_utilization=0.9,
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_Engine_Configuration]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_CUDA_Environment]]
