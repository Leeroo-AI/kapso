# LLM.generate

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::Inference]], [[domain::Generation]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:30 GMT]]
|}

== Overview ==

Concrete tool for executing batch text generation with automatic batching, memory management, and progress tracking.

=== Description ===

`LLM.generate()` is the primary method for offline batch inference in vLLM. It accepts multiple prompts and returns completions for all of them, automatically handling:

* Dynamic batching across prompts of varying lengths
* Memory allocation via PagedAttention
* Progress display with optional tqdm
* LoRA adapter application per-request
* Priority scheduling (when enabled)

The method blocks until all prompts are processed and returns results in the same order as inputs.

=== Usage ===

Use `LLM.generate()` for:
* Batch processing multiple prompts
* Offline evaluation and benchmarking
* Data generation pipelines
* Any scenario where latency is less critical than throughput

For streaming or online serving, use the async API instead.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm]
* '''File:''' vllm/entrypoints/llm.py

=== Signature ===
<syntaxhighlight lang="python">
def generate(
    self,
    prompts: PromptType | Sequence[PromptType],
    sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
    *,
    use_tqdm: bool | Callable[..., tqdm] = True,
    lora_request: list[LoRARequest] | LoRARequest | None = None,
    priority: list[int] | None = None,
) -> list[RequestOutput]:
    """Generates the completions for the input prompts.

    This class automatically batches the given prompts, considering
    the memory constraint. For the best performance, put all of your
    prompts into a single list and pass it to this method.

    Args:
        prompts: The prompts to the LLM. See PromptType for formats.
        sampling_params: Sampling parameters. If None, uses defaults.
            Can be a single value (applied to all) or per-prompt list.
        use_tqdm: Show progress bar. Can be True, False, or custom tqdm.
        lora_request: LoRA adapter(s) to use for generation.
        priority: Request priorities (when priority scheduling enabled).

    Returns:
        List of RequestOutput in same order as input prompts.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| prompts || PromptType/Sequence || Yes || Single prompt or list of prompts
|-
| sampling_params || SamplingParams || No || Generation parameters (uses defaults if None)
|-
| use_tqdm || bool/Callable || No || Progress bar control (default: True)
|-
| lora_request || LoRARequest || No || LoRA adapter(s) to apply
|-
| priority || list[int] || No || Request priorities (requires priority scheduler)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| outputs || list[RequestOutput] || Generated completions, same order as prompts
|}

== Usage Examples ==

=== Basic Batch Generation ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

prompts = [
    "What is the capital of France?",
    "Explain photosynthesis in one sentence.",
    "Write a haiku about programming.",
]

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=100,
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}\n")
</syntaxhighlight>

=== Per-Prompt Sampling Parameters ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

prompts = [
    "Give me a factual answer: What is 2+2?",   # Greedy
    "Write a creative story about a robot.",     # High temp
    "Summarize the following text...",           # Medium temp
]

# Different params per prompt
params = [
    SamplingParams(temperature=0, max_tokens=50),      # Greedy
    SamplingParams(temperature=0.9, max_tokens=500),   # Creative
    SamplingParams(temperature=0.3, max_tokens=200),   # Balanced
]

outputs = llm.generate(prompts, params)
</syntaxhighlight>

=== Large Batch with Progress ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

# Generate 1000 completions
prompts = [f"Question {i}: What is {i} + {i}?" for i in range(1000)]
params = SamplingParams(temperature=0, max_tokens=20)

# Progress bar enabled by default
outputs = llm.generate(prompts, params)

# Or disable progress bar
outputs = llm.generate(prompts, params, use_tqdm=False)
</syntaxhighlight>

=== With LoRA Adapter ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_lora=True,
)

prompts = ["Translate to French: Hello, world!"]
params = SamplingParams(temperature=0.7, max_tokens=50)

lora = LoRARequest(
    lora_name="translation-lora",
    lora_int_id=1,
    lora_path="/path/to/translation-lora",
)

outputs = llm.generate(prompts, params, lora_request=lora)
</syntaxhighlight>

=== Multiple Outputs per Prompt ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

prompts = ["Write a creative opening line for a novel."]

# Generate 5 different completions per prompt
params = SamplingParams(
    n=5,                    # 5 outputs per prompt
    temperature=0.9,
    max_tokens=50,
)

outputs = llm.generate(prompts, params)

for i, completion in enumerate(outputs[0].outputs):
    print(f"Option {i+1}: {completion.text}")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_Batch_Generation]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_CUDA_Environment]]
