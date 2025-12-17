= EngineArgs Multimodal API =

{{Metadata
| Knowledge Sources = vllm/engine/arg_utils.py, vllm/entrypoints/llm.py, examples
| Domains = API Reference, Model Configuration, Multimodal Systems
| Last Updated = 2025-12-17
}}

== Overview ==

The '''EngineArgs Multimodal API''' implements the VLM Configuration Principle by providing a dataclass-based interface for specifying all parameters needed to initialize a vision-language model in vLLM. This API consolidates model configuration, resource allocation, and multimodal processing settings into a single cohesive interface.

== Code Reference ==

=== Source Location ===

<syntaxhighlight lang="python">
# Primary implementation
vllm/engine/arg_utils.py: class EngineArgs
vllm/entrypoints/llm.py: class LLM.__init__
</syntaxhighlight>

=== Signature ===

<syntaxhighlight lang="python">
@dataclass
class EngineArgs:
    model: str
    tokenizer: Optional[str] = None
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    trust_remote_code: bool = False
    mm_processor_kwargs: Optional[Dict[str, Any]] = None
    limit_mm_per_prompt: Optional[Dict[str, int]] = None
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.9
    # ... additional parameters
</syntaxhighlight>

=== Import ===

<syntaxhighlight lang="python">
from vllm import LLM, EngineArgs
</syntaxhighlight>

== Description ==

The `EngineArgs` class and `LLM` constructor implement multimodal configuration through several key parameters:

* '''model''': HuggingFace model identifier (e.g., "llava-hf/llava-1.5-7b-hf")
* '''trust_remote_code''': Enable execution of custom model code
* '''mm_processor_kwargs''': Dictionary of processor overrides (model-specific)
* '''limit_mm_per_prompt''': Dictionary specifying maximum inputs per modality
* '''max_model_len''': Maximum sequence length for the model
* '''max_num_seqs''': Maximum number of sequences to process in parallel

These parameters are validated and used to create a `VllmConfig` object that controls all aspects of model initialization and runtime behavior.

== I/O Contract ==

=== Input Parameters ===

{| class="wikitable"
! Parameter !! Type !! Description
|-
| model || str || HuggingFace model path or name
|-
| trust_remote_code || bool || Allow remote code execution (default: False)
|-
| mm_processor_kwargs || Dict[str, Any] || Processor-specific overrides
|-
| limit_mm_per_prompt || Dict[str, int] || Max inputs per modality (e.g., {"image": 1})
|-
| max_model_len || int || Maximum sequence length
|-
| tensor_parallel_size || int || Number of GPUs for tensor parallelism
|-
| gpu_memory_utilization || float || Fraction of GPU memory to use (0-1)
|}

=== Output ===

Returns an initialized `LLM` object configured for multimodal inference.

== Usage Examples ==

=== Basic VLM Configuration ===

<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

# Initialize LLaVA-1.5 with basic configuration
llm = LLM(
    model="llava-hf/llava-1.5-7b-hf",
    trust_remote_code=False,
    max_model_len=4096,
    limit_mm_per_prompt={"image": 1},
)
</syntaxhighlight>

=== Advanced Configuration with Processor Kwargs ===

<syntaxhighlight lang="python">
from vllm import LLM

# Configure Phi-3-Vision with custom processor settings
llm = LLM(
    model="microsoft/Phi-3.5-vision-instruct",
    trust_remote_code=True,
    max_model_len=4096,
    max_num_seqs=2,
    mm_processor_kwargs={"num_crops": 16},  # Controls image tiling
    limit_mm_per_prompt={"image": 1},
)
</syntaxhighlight>

=== Multi-GPU Configuration ===

<syntaxhighlight lang="python">
from vllm import LLM

# Configure large VLM with tensor parallelism
llm = LLM(
    model="llava-hf/llava-v1.6-34b-hf",
    tensor_parallel_size=4,
    max_model_len=8192,
    gpu_memory_utilization=0.85,
    limit_mm_per_prompt={"image": 1},
)
</syntaxhighlight>

=== Video Model Configuration ===

<syntaxhighlight lang="python">
from vllm import LLM

# Configure video-language model
llm = LLM(
    model="llava-hf/LLaVA-NeXT-Video-7B-hf",
    max_model_len=8192,
    max_num_seqs=2,
    limit_mm_per_prompt={"video": 1},
    mm_processor_kwargs={"fps": 1.0},  # Video frame sampling rate
)
</syntaxhighlight>

=== Using EngineArgs Dataclass ===

<syntaxhighlight lang="python">
from vllm import LLM, EngineArgs
from dataclasses import asdict

# Create engine args separately
engine_args = EngineArgs(
    model="Qwen/Qwen2-VL-7B-Instruct",
    max_model_len=4096,
    max_num_seqs=5,
    mm_processor_kwargs={
        "min_pixels": 28 * 28,
        "max_pixels": 1280 * 28 * 28,
    },
    limit_mm_per_prompt={"image": 1},
)

# Initialize LLM from engine args
llm = LLM(**asdict(engine_args))
</syntaxhighlight>

== Related Pages ==

* [[implements::Principle:vllm-project_vllm_VLM_Configuration_Principle]]
* [[next_step::vllm-project_vllm_Image_Loading_API]]
* [[uses::vllm-project_vllm_LLM_Class]]
