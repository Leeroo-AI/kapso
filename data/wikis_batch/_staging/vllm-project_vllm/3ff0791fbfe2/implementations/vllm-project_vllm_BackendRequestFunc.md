{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Benchmarking]], [[domain::HTTP]], [[domain::Testing]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==
Unified async HTTP request handlers for benchmarking multiple LLM serving backends.

=== Description ===
This module provides backend-agnostic async request functions for benchmarking LLM serving systems. It includes standardized request/response data classes (RequestFuncInput, RequestFuncOutput) and async handlers for TGI, TensorRT-LLM, DeepSpeed-MII, and OpenAI-compatible APIs (completions, chat completions, audio). Each handler tracks detailed timing metrics including time-to-first-token (TTFT), inter-token latencies (ITL), and tokens-per-output-token (TPOT). The module also provides utilities for loading models and tokenizers.

=== Usage ===
Use this module when building benchmarking tools that need to test against multiple LLM serving backends with consistent timing measurements. It enables fair performance comparisons by providing standardized interfaces across vLLM, TGI, TensorRT-LLM, and other backends.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/backend_request_func.py#L1-L657 benchmarks/backend_request_func.py]
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

=== Import ===
<syntaxhighlight lang="python">
from benchmarks.backend_request_func import (
    RequestFuncInput,
    RequestFuncOutput,
    ASYNC_REQUEST_FUNCS,
    async_request_openai_completions,
    async_request_openai_chat_completions,
    get_tokenizer,
)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| request_func_input || RequestFuncInput || Yes || Contains prompt, API URL, model info, and request parameters
|-
| pbar || tqdm \| None || No || Optional progress bar to update on completion
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| output || RequestFuncOutput || Contains generated text, timing metrics (TTFT, ITL, latency), success status, and error info
|}

== Usage Examples ==
<syntaxhighlight lang="python">
import asyncio
from benchmarks.backend_request_func import (
    RequestFuncInput,
    async_request_openai_completions,
    ASYNC_REQUEST_FUNCS,
)

# Example 1: OpenAI-compatible completions API
async def benchmark_vllm():
    request_input = RequestFuncInput(
        prompt="What is machine learning?",
        api_url="http://localhost:8000/v1/completions",
        prompt_len=5,
        output_len=100,
        model="meta-llama/Llama-2-7b-hf",
        ignore_eos=False,
    )

    output = await async_request_openai_completions(request_input)

    if output.success:
        print(f"Generated: {output.generated_text}")
        print(f"TTFT: {output.ttft:.3f}s")
        print(f"Total latency: {output.latency:.3f}s")
        print(f"Output tokens: {output.output_tokens}")
    else:
        print(f"Error: {output.error}")

# Example 2: Using the backend registry
async def benchmark_any_backend(backend_name: str):
    request_func = ASYNC_REQUEST_FUNCS[backend_name]

    request_input = RequestFuncInput(
        prompt="Explain quantum computing",
        api_url="http://localhost:8000/generate_stream",
        prompt_len=3,
        output_len=50,
        model="gpt2",
    )

    output = await request_func(request_input)
    return output

# Example 3: Batch benchmarking with progress bar
from tqdm.asyncio import tqdm

async def batch_benchmark():
    prompts = [
        "What is AI?",
        "Explain neural networks",
        "How does deep learning work?",
    ]

    tasks = []
    pbar = tqdm(total=len(prompts))

    for prompt in prompts:
        request_input = RequestFuncInput(
            prompt=prompt,
            api_url="http://localhost:8000/v1/completions",
            prompt_len=len(prompt.split()),
            output_len=100,
            model="llama-2-7b",
        )
        tasks.append(async_request_openai_completions(request_input, pbar))

    outputs = await asyncio.gather(*tasks)
    pbar.close()

    # Calculate average metrics
    avg_ttft = sum(o.ttft for o in outputs) / len(outputs)
    avg_latency = sum(o.latency for o in outputs) / len(outputs)
    print(f"Average TTFT: {avg_ttft:.3f}s")
    print(f"Average latency: {avg_latency:.3f}s")

# Run examples
asyncio.run(benchmark_vllm())
asyncio.run(benchmark_any_backend("vllm"))
asyncio.run(batch_benchmark())
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_CUDA_Environment]]
