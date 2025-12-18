{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Paper|vLLM: Easy, Fast, and Cheap LLM Serving|https://arxiv.org/abs/2309.06180]]
|-
! Domains
| [[domain::NLP]], [[domain::Inference]], [[domain::Batch_Processing]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

Concrete tool for executing batch text generation on multiple prompts with automatic batching, memory management, and optional LoRA adapter support.

=== Description ===

`LLM.generate()` is the primary method for running inference in vLLM's offline mode. It accepts a batch of prompts, automatically manages memory and scheduling, and returns generated outputs. Key features:

- **Automatic Batching:** Processes prompts together for GPU efficiency
- **Memory Management:** Respects GPU memory limits via PagedAttention
- **Continuous Batching:** Adds/removes sequences dynamically
- **Progress Tracking:** Optional tqdm progress bar
- **LoRA Support:** Can apply LoRA adapters per-request

=== Usage ===

Use `LLM.generate()` for:
- Batch processing of text generation tasks
- Offline evaluation and benchmarking
- Data processing pipelines with LLMs
- Development and testing of prompts

For streaming or online serving, use `AsyncLLMEngine` or `vllm serve` instead.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project/vllm]
* '''File:''' vllm/entrypoints/llm.py
* '''Lines:''' L365-434

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
    """
    Generates completions for the input prompts.

    Args:
        prompts: Input prompts. Can be a single prompt or sequence.
            See PromptType for supported formats (str, dict, token IDs).
        sampling_params: Sampling configuration. If None, uses defaults.
            Can be single (applied to all) or list (per-prompt).
        use_tqdm: Show progress bar. True=default, False=disabled,
            or pass custom tqdm callable.
        lora_request: LoRA adapter(s) to apply. Single or list.
        priority: Request priorities for priority scheduling.

    Returns:
        List of RequestOutput objects in same order as input prompts.
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
| prompts || PromptType | Sequence[PromptType] || Yes || Input prompts (strings, dicts, or token IDs)
|-
| sampling_params || SamplingParams | Sequence[SamplingParams] || No || Generation parameters (default: model-specific)
|-
| use_tqdm || bool | Callable || No || Progress bar control (default: True)
|-
| lora_request || LoRARequest | list[LoRARequest] || No || LoRA adapter(s) to use
|-
| priority || list[int] || No || Request priorities (if priority scheduling enabled)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| outputs || list[RequestOutput] || Generated completions, one per input prompt
|}

== Usage Examples ==

=== Basic Batch Generation ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

prompts = [
    "What is machine learning?",
    "Explain the theory of relativity.",
    "Write a poem about the ocean.",
]

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=200,
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}\n")
</syntaxhighlight>

=== Different Params Per Prompt ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

prompts = [
    "Summarize this article:",     # Factual - low temp
    "Write a creative story:",      # Creative - high temp
    "What is 2 + 2?",               # Deterministic - greedy
]

# Different params for each prompt
params = [
    SamplingParams(temperature=0.3, max_tokens=100),
    SamplingParams(temperature=0.9, max_tokens=300, top_p=0.95),
    SamplingParams(temperature=0, max_tokens=10),
]

outputs = llm.generate(prompts, params)
</syntaxhighlight>

=== Large-Scale Batch Processing ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.9,
)

# Load 10,000 prompts
with open("prompts.txt") as f:
    prompts = [line.strip() for line in f]

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=256,
    stop=["\n\n"],
)

# Process all at once - vLLM handles batching internally
outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

# Save results
with open("outputs.txt", "w") as f:
    for output in outputs:
        f.write(output.outputs[0].text + "\n")
</syntaxhighlight>

=== Silent Processing (No Progress Bar) ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

# Disable progress bar for scripts/pipelines
outputs = llm.generate(
    prompts=["Hello, world!"],
    sampling_params=SamplingParams(max_tokens=50),
    use_tqdm=False,
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_Batch_Generation]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
