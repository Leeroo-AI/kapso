= LLM Class =

== Metadata ==
{| class="wikitable"
|-
! Field !! Value
|-
| Knowledge Sources || vllm/entrypoints/llm.py
|-
| Domains || API Design, Inference Engine, Distributed Systems
|-
| Last Updated || 2025-12-17
|}

== Overview ==
The <code>LLM</code> class is vLLM's primary interface for offline inference. It provides a high-level API for generating text from prompts, managing a language model across potentially multiple GPUs, and handling tokenization. The class supports both single-instance and distributed data parallel execution.

== Description ==
<code>LLM</code> encapsulates the complete inference pipeline:

* '''Tokenization''': Converts text prompts to token IDs
* '''Engine Management''': Orchestrates the underlying LLMEngine
* '''GPU Memory Management''': Allocates and manages KV cache
* '''Batching''': Intelligently groups requests for efficient processing
* '''Distributed Support''': Automatically detects and configures data parallel mode

The class provides a simple, synchronous API that hides the complexity of distributed execution, model loading, and resource management.

== Code Reference ==

=== Source Location ===
* '''File''': <code>/tmp/praxium_repo_583nq7ea/vllm/entrypoints/llm.py</code>
* '''Lines''': 92-187 (class definition and docstring)

=== Class Signature ===
<syntaxhighlight lang="python">
class LLM:
    """An LLM for generating texts from given prompts and sampling parameters.

    This class includes a tokenizer, a language model (possibly distributed
    across multiple GPUs), and GPU memory space allocated for intermediate
    states (aka KV cache). Given a batch of prompts and sampling parameters,
    this class generates texts from the model, using an intelligent batching
    mechanism and efficient memory management.
    """

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
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: str | None = None,
        revision: str | None = None,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
        # ... many more parameters
        **kwargs
    ):
        # Initialization logic
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
</syntaxhighlight>

== I/O Contract ==

=== Constructor Parameters ===
{| class="wikitable"
|-
! Parameter !! Type !! Default !! Description
|-
| model || str || required || Model name or path
|-
| tokenizer || str || None || Tokenizer name or path (defaults to model)
|-
| tokenizer_mode || str || "auto" || Tokenizer mode ("auto", "slow", "mistral")
|-
| trust_remote_code || bool || False || Trust remote code when loading model
|-
| tensor_parallel_size || int || 1 || Number of GPUs for tensor parallelism
|-
| dtype || str || "auto" || Data type ("auto", "float16", "bfloat16", "float32")
|-
| quantization || str || None || Quantization method ("awq", "gptq", "fp8")
|-
| gpu_memory_utilization || float || 0.9 || Fraction of GPU memory to use (0.0-1.0)
|-
| max_model_len || int || None || Maximum sequence length
|-
| max_num_seqs || int || 256 || Maximum concurrent sequences
|-
| enforce_eager || bool || False || Disable CUDA graphs, use eager execution
|-
| enable_expert_parallel || bool || False || Enable expert parallelism for MoE models
|}

=== Key Methods ===
{| class="wikitable"
|-
! Method !! Returns !! Description
|-
| generate() || list[RequestOutput] || Generate completions for prompts
|-
| chat() || list[RequestOutput] || Generate chat completions with conversation history
|-
| encode() || list[EmbeddingRequestOutput] || Generate embeddings for prompts
|-
| score() || list[ScoringRequestOutput] || Score prompt-completion pairs
|-
| get_metrics() || dict[str, Metric] || Retrieve engine metrics
|}

== Usage Examples ==

=== Basic Text Generation ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

# Initialize LLM
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

# Create prompts
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "The future of AI is"
]

# Configure sampling
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100
)

# Generate
outputs = llm.generate(prompts, sampling_params)

# Process outputs
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated: {generated_text!r}\n")
</syntaxhighlight>

=== Data Parallel Initialization ===
<syntaxhighlight lang="python">
import os
from vllm import LLM, SamplingParams

# Worker entry point (called in each process)
def worker_main(rank, size, prompts):
    # Set DP environment variables
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(size)
    os.environ["VLLM_DP_MASTER_IP"] = "127.0.0.1"
    os.environ["VLLM_DP_MASTER_PORT"] = "29500"

    # Initialize LLM - automatically detects DP mode
    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel_size=2
    )

    # Each worker processes its partition
    outputs = llm.generate(prompts, SamplingParams(max_tokens=100))
    return outputs
</syntaxhighlight>

=== With Tensor Parallelism ===
<syntaxhighlight lang="python">
from vllm import LLM

# Split model across 4 GPUs
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    dtype="bfloat16",
    gpu_memory_utilization=0.95
)

outputs = llm.generate(prompts, sampling_params)
</syntaxhighlight>

=== With Expert Parallelism (MoE Models) ===
<syntaxhighlight lang="python">
from vllm import LLM

# MoE model with expert parallelism
llm = LLM(
    model="ibm-research/PowerMoE-3b",
    tensor_parallel_size=2,
    enable_expert_parallel=True,
    expert_placement_strategy="round_robin",
    trust_remote_code=True
)

outputs = llm.generate(prompts, sampling_params)
</syntaxhighlight>

=== Chat Completion ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

# Chat messages format
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]

outputs = llm.chat(
    messages=[messages],
    sampling_params=SamplingParams(temperature=0.7, max_tokens=100)
)

print(outputs[0].outputs[0].text)
</syntaxhighlight>

=== Generate Embeddings ===
<syntaxhighlight lang="python">
from vllm import LLM

# Initialize with embedding model
llm = LLM(model="intfloat/e5-mistral-7b-instruct")

prompts = [
    "This is a test sentence.",
    "Another example sentence."
]

# Generate embeddings
outputs = llm.encode(prompts)

for output in outputs:
    embedding = output.outputs.embedding
    print(f"Embedding shape: {len(embedding)}")
</syntaxhighlight>

=== With Custom Configuration ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=2,
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
    max_model_len=4096,
    max_num_seqs=64,
    enforce_eager=False,
    trust_remote_code=False,
    disable_custom_all_reduce=False
)

# Custom sampling parameters per prompt
sampling_params_1 = SamplingParams(temperature=0.7, max_tokens=50)
sampling_params_2 = SamplingParams(temperature=0.9, max_tokens=100)

# Generate with different parameters
outputs = llm.generate(
    prompts=["Prompt 1", "Prompt 2"],
    sampling_params=[sampling_params_1, sampling_params_2]
)
</syntaxhighlight>

=== Retrieve Metrics ===
<syntaxhighlight lang="python">
from vllm import LLM

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

# Generate some outputs
outputs = llm.generate(prompts, sampling_params)

# Get engine metrics
metrics = llm.get_metrics()

print(f"Total requests: {metrics.get('num_requests_running', 0)}")
print(f"GPU KV cache usage: {metrics.get('gpu_cache_usage', 0)}")
</syntaxhighlight>

== Implementation Details ==

=== Automatic DP Detection ===
The LLM class automatically detects data parallel mode:
<syntaxhighlight lang="python">
# Inside __init__
if os.environ.get("VLLM_DP_SIZE"):
    # DP mode detected - configure accordingly
    self.data_parallel_size = int(os.environ["VLLM_DP_SIZE"])
    self.data_parallel_rank = int(os.environ["VLLM_DP_RANK"])
    # ... setup DP communication
</syntaxhighlight>

=== Engine Creation ===
The LLM class creates the underlying engine:
<syntaxhighlight lang="python">
# Simplified initialization flow
def __init__(self, model, **kwargs):
    # Create engine arguments
    engine_args = EngineArgs(model=model, **kwargs)

    # Create engine
    self.llm_engine = LLMEngine.from_engine_args(engine_args)

    # Setup tokenizer
    if not skip_tokenizer_init:
        self.tokenizer = self._init_tokenizer()
</syntaxhighlight>

=== Synchronous API ===
All methods are synchronous and blocking:
* <code>generate()</code> blocks until all completions finish
* Simplifies usage for offline inference
* No async/await complexity
* For async, use AsyncLLMEngine instead

=== Batch Processing ===
The engine automatically batches multiple prompts:
* Groups prompts with similar characteristics
* Optimizes GPU utilization
* Handles variable-length sequences efficiently

== Best Practices ==

# '''Memory Planning''': Set <code>gpu_memory_utilization</code> based on workload (0.8-0.95)
# '''Tensor Parallelism''': Use TP for large models that don't fit in single GPU
# '''Data Parallelism''': Use DP for high throughput with many concurrent requests
# '''Eager Mode''': Use <code>enforce_eager=True</code> for debugging or dynamic shapes
# '''Quantization''': Use quantization for memory-constrained scenarios

== Common Issues ==

=== OOM Errors ===
* Reduce <code>gpu_memory_utilization</code>
* Reduce <code>max_num_seqs</code>
* Reduce <code>max_model_len</code>
* Use quantization

=== Slow Initialization ===
* Model download time (first run)
* Large model weight loading
* CUDA graph compilation
* Use <code>enforce_eager=True</code> to skip graph compilation

=== Incorrect Outputs ===
* Verify <code>trust_remote_code</code> for custom models
* Check <code>dtype</code> matches model requirements
* Validate tokenizer configuration

== Related Pages ==
* [[implements::Principle:vllm-project_vllm_LLM_distributed]] - Distributed initialization principle
* [[related_to::vllm-project_vllm_LLM_generate_dp]] - Generate method in DP mode
* [[related_to::SamplingParams]] - Sampling configuration
* [[related_to::LLMEngine]] - Underlying engine implementation

== See Also ==
* vllm/entrypoints/llm.py - Full implementation
* vllm/v1/engine/llm_engine.py - Engine core
* vLLM Quickstart Documentation
