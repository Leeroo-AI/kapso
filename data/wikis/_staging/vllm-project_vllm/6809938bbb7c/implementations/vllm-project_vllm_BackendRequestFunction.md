{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::HTTP]], [[domain::AsyncIO]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Async request handlers for various LLM serving backends used in benchmarking throughput and latency.

=== Description ===
This module provides asynchronous request functions for benchmarking multiple LLM serving backends including vLLM, TGI, TensorRT-LLM, DeepSpeed-MII, and others. Each request handler implements streaming responses with detailed latency metrics including time-to-first-token (TTFT), inter-token latency (ITL), and total latency. The module is designed to work without requiring vLLM installation, making it portable for comparative benchmarking.

The request functions support various API formats (OpenAI Completions, Chat Completions, Audio transcription) and track granular performance metrics. All handlers use aiohttp for async HTTP communication with a 6-hour timeout. The module also includes tokenizer utilities with support for ModelScope and Mistral tokenizers.

=== Usage ===
Use this module when running throughput/latency benchmarks across different LLM serving backends, especially for API compatibility testing and performance comparison studies.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/backend_request_func.py benchmarks/backend_request_func.py]
* '''Lines:''' 1-657

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    model_name: str | None = None
    logprobs: int | None = None
    extra_body: dict | None = None
    multi_modal_content: dict | list[dict] | None = None
    ignore_eos: bool = False
    language: str | None = None
    request_id: str | None = None

@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0
    itl: list[float] = field(default_factory=list)
    tpot: float = 0.0
    prompt_len: int = 0
    error: str = ""

async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: tqdm | None = None,
) -> RequestFuncOutput
</syntaxhighlight>

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Used by benchmark_serving.py and other benchmarking scripts
python benchmarks/benchmark_serving.py --backend vllm --model MODEL_NAME
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| request_func_input || RequestFuncInput || Contains prompt, API URL, model info, and request parameters
|-
| pbar || tqdm (optional) || Progress bar for tracking request completion
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| output || RequestFuncOutput || Contains generated text, latency metrics (TTFT, ITL, total), success status, and errors
|}

== Usage Examples ==

<syntaxhighlight lang="python">
import asyncio
from benchmarks.backend_request_func import (
    RequestFuncInput,
    async_request_openai_completions,
    get_tokenizer
)

# Create request input
request_input = RequestFuncInput(
    prompt="What is the capital of France?",
    api_url="http://localhost:8000/v1/completions",
    prompt_len=10,
    output_len=50,
    model="meta-llama/Llama-2-7b-hf",
    ignore_eos=False
)

# Execute async request
async def benchmark():
    output = await async_request_openai_completions(request_input)
    print(f"TTFT: {output.ttft:.3f}s")
    print(f"Total latency: {output.latency:.3f}s")
    print(f"Tokens generated: {output.output_tokens}")
    print(f"Generated text: {output.generated_text[:100]}")

asyncio.run(benchmark())

# Get tokenizer for prompt counting
tokenizer = get_tokenizer("meta-llama/Llama-2-7b-hf")
prompt_tokens = len(tokenizer.encode("Your prompt here"))
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
