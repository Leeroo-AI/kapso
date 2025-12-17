= LLM Multimodal Initialization API =

{{Metadata
| Knowledge Sources = vllm/entrypoints/llm.py, initialization flow analysis
| Domains = API Reference, Engine Initialization, Resource Management
| Last Updated = 2025-12-17
}}

== Overview ==

The '''LLM Multimodal Initialization API''' implements the VLM Engine Initialization Principle by providing the `LLM` class constructor that handles all aspects of vision-language model initialization. This API is the primary entry point for creating inference-ready VLM instances.

== Code Reference ==

=== Source Location ===

<syntaxhighlight lang="python">
# Main LLM class
vllm/entrypoints/llm.py: class LLM.__init__

# Engine initialization
vllm/v1/engine/llm_engine.py: class LLMEngine.from_engine_args

# Configuration
vllm/config/__init__.py: VllmConfig
</syntaxhighlight>

=== Signature ===

<syntaxhighlight lang="python">
class LLM:
    def __init__(
        self,
        model: str,
        *,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
        limit_mm_per_prompt: Optional[Dict[str, int]] = None,
        max_model_len: Optional[int] = None,
        max_num_seqs: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        # Initialization logic
</syntaxhighlight>

=== Import ===

<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
</syntaxhighlight>

== Description ==

The `LLM.__init__` method performs the following initialization steps:

1. '''Parameter Validation''': Validates model path, memory settings, and parallelism configuration
2. '''Engine Args Creation''': Constructs `EngineArgs` with all configuration parameters
3. '''Engine Instantiation''': Creates `LLMEngine` instance via `from_engine_args()`
4. '''Processor Setup''': Initializes tokenizer and multimodal processors
5. '''Memory Allocation''': Allocates GPU memory pools for KV cache and features
6. '''Model Loading''': Loads model weights with appropriate device placement
7. '''Warmup''': Performs optional warmup to prepare for inference

The initialization is synchronous and blocks until the model is ready for inference.

== I/O Contract ==

=== Input Parameters ===

{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| model || str || Required || HuggingFace model identifier
|-
| trust_remote_code || bool || False || Enable custom model code
|-
| tensor_parallel_size || int || 1 || Number of GPUs for tensor parallelism
|-
| dtype || str || "auto" || Model weight data type (auto, float16, bfloat16)
|-
| gpu_memory_utilization || float || 0.9 || Fraction of GPU memory to use
|-
| mm_processor_kwargs || Dict || None || Processor overrides
|-
| limit_mm_per_prompt || Dict || None || Max multimodal inputs per modality
|-
| max_model_len || int || None || Maximum sequence length
|-
| max_num_seqs || int || None || Maximum concurrent sequences
|}

=== Output ===

Returns an initialized `LLM` instance ready for inference via `generate()`, `chat()`, or other methods.

== Usage Examples ==

=== Basic Initialization ===

<syntaxhighlight lang="python">
from vllm import LLM

# Initialize LLaVA-1.5 with defaults
llm = LLM(model="llava-hf/llava-1.5-7b-hf")

# Model is now ready for inference
print("Model loaded successfully")
print(f"Supported tasks: {llm.supported_tasks}")
</syntaxhighlight>

=== Initialization with Memory Configuration ===

<syntaxhighlight lang="python">
from vllm import LLM

# Configure memory usage for large models
llm = LLM(
    model="llava-hf/llava-v1.6-34b-hf",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.85,
    max_model_len=8192,
    dtype="bfloat16"
)
</syntaxhighlight>

=== Initialization with Processor Configuration ===

<syntaxhighlight lang="python">
from vllm import LLM

# Configure image processor for Phi-3-Vision
llm = LLM(
    model="microsoft/Phi-3.5-vision-instruct",
    trust_remote_code=True,
    max_model_len=4096,
    max_num_seqs=2,
    mm_processor_kwargs={
        "num_crops": 16  # Controls image tiling/resolution
    },
    limit_mm_per_prompt={"image": 1}
)

# Processor is configured and ready
</syntaxhighlight>

=== Initialization with Multiple Modalities ===

<syntaxhighlight lang="python">
from vllm import LLM

# Initialize model supporting images and videos
llm = LLM(
    model="openbmb/MiniCPM-V-2_6",
    trust_remote_code=True,
    max_model_len=4096,
    limit_mm_per_prompt={
        "image": 1,
        "video": 1
    }
)
</syntaxhighlight>

=== Initialization with Quantization ===

<syntaxhighlight lang="python">
from vllm import LLM

# Load quantized model for reduced memory usage
llm = LLM(
    model="llava-hf/llava-1.5-7b-hf",
    quantization="awq",
    max_model_len=4096,
    gpu_memory_utilization=0.8
)
</syntaxhighlight>

=== Checking Initialization Status ===

<syntaxhighlight lang="python">
from vllm import LLM

llm = LLM(model="llava-hf/llava-1.5-7b-hf")

# Access model configuration
print(f"Model config: {llm.model_config}")
print(f"Max model length: {llm.model_config.max_model_len}")
print(f"Is multimodal: {llm.model_config.is_multimodal_model}")

# Access tokenizer
tokenizer = llm.get_tokenizer()
print(f"Tokenizer vocab size: {len(tokenizer)}")

# Check supported tasks
print(f"Supported tasks: {llm.supported_tasks}")
</syntaxhighlight>

=== Initialization with Error Handling ===

<syntaxhighlight lang="python">
from vllm import LLM
import torch

try:
    llm = LLM(
        model="llava-hf/llava-1.5-7b-hf",
        tensor_parallel_size=2,
        max_model_len=4096
    )
    print("Model initialized successfully")
except torch.cuda.OutOfMemoryError:
    print("Out of memory. Try reducing max_model_len or gpu_memory_utilization")
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Initialization failed: {e}")
</syntaxhighlight>

=== Initialization for Video Models ===

<syntaxhighlight lang="python">
from vllm import LLM

# Initialize video-language model
llm = LLM(
    model="llava-hf/LLaVA-NeXT-Video-7B-hf",
    max_model_len=8192,
    max_num_seqs=2,
    limit_mm_per_prompt={"video": 1},
    mm_processor_kwargs={
        "fps": 1.0  # Frame sampling rate for videos
    }
)
</syntaxhighlight>

=== Accessing Input Processor ===

<syntaxhighlight lang="python">
from vllm import LLM

llm = LLM(model="llava-hf/llava-1.5-7b-hf")

# Access the input processor
print(f"Input processor: {llm.input_processor}")

# Check multimodal cache status
print(f"Model config multimodal: {llm.model_config.is_multimodal_model}")
</syntaxhighlight>

== Initialization Flow ==

The initialization process follows these stages:

1. '''Argument Validation''': Check parameter compatibility and constraints
2. '''Config Creation''': Build `VllmConfig` from parameters
3. '''Device Setup''': Initialize GPUs and distributed communication
4. '''Model Loading''': Load weights with device mapping
5. '''Processor Setup''': Initialize tokenizer and multimodal processors
6. '''Memory Allocation''': Allocate KV cache and feature storage
7. '''Pipeline Creation''': Set up input/output processing pipelines
8. '''Readiness''': Mark engine as ready for inference

== Related Pages ==

* [[implements::Principle:vllm-project_vllm_VLM_Engine_Initialization_Principle]]
* [[next_step::vllm-project_vllm_LLM_Generate_Multimodal_API]]
* [[uses::vllm-project_vllm_EngineArgs_Multimodal_API]]
