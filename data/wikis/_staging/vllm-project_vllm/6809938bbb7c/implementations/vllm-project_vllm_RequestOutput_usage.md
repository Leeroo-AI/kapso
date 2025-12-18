{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::NLP]], [[domain::Output_Processing]], [[domain::Data_Structures]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

Concrete tool for accessing generated text, token IDs, log probabilities, and metadata from vLLM inference results.

=== Description ===

`RequestOutput` is the data class returned by `LLM.generate()` containing all information about a completed generation request. It wraps one or more `CompletionOutput` objects (when `n > 1` in sampling params) and provides access to:

- Generated text and token IDs
- Log probabilities (if requested)
- Finish reason and stop conditions
- Request metadata and metrics
- Original prompt information

=== Usage ===

Use `RequestOutput` after calling `LLM.generate()` to:
- Extract generated text for downstream processing
- Analyze token-level log probabilities
- Check finish reasons for quality control
- Access performance metrics for benchmarking
- Handle multiple completions per prompt (n > 1)

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project/vllm]
* '''File:''' vllm/outputs.py
* '''Lines:''' L84-191

=== Signature ===
<syntaxhighlight lang="python">
class RequestOutput:
    """The output data of a completion request to the LLM."""

    request_id: str
    """Unique identifier for this request."""

    prompt: str | None
    """Original prompt string."""

    prompt_token_ids: list[int] | None
    """Token IDs of the prompt."""

    prompt_logprobs: PromptLogprobs | None
    """Log probabilities of prompt tokens (if requested)."""

    outputs: list[CompletionOutput]
    """Generated completions (one per n in SamplingParams)."""

    finished: bool
    """Whether generation is complete."""

    metrics: RequestMetrics | None
    """Performance metrics for this request."""

    lora_request: LoRARequest | None
    """LoRA adapter used (if any)."""

    num_cached_tokens: int | None
    """Number of tokens served from prefix cache."""


class CompletionOutput:
    """Single completion output."""

    index: int
    """Index when n > 1."""

    text: str
    """Generated text."""

    token_ids: Sequence[int]
    """Generated token IDs."""

    cumulative_logprob: float | None
    """Sum of log probabilities."""

    logprobs: SampleLogprobs | None
    """Per-token log probabilities."""

    finish_reason: str | None
    """Why generation stopped: "stop", "length", or None."""

    stop_reason: int | str | None
    """Specific stop string/token that triggered stop."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm import RequestOutput
from vllm.outputs import CompletionOutput
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| RequestOutput || RequestOutput || Yes || Output object from LLM.generate()
|}

=== Outputs (Accessible Fields) ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| outputs[i].text || str || Generated text for completion i
|-
| outputs[i].token_ids || list[int] || Token IDs of generated text
|-
| outputs[i].logprobs || list[dict] || Per-token log probabilities
|-
| outputs[i].finish_reason || str || "stop", "length", or None
|-
| prompt || str || Original input prompt
|-
| num_cached_tokens || int || Prefix cache hits
|}

== Usage Examples ==

=== Basic Text Extraction ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

prompts = ["What is AI?", "Explain Python."]
outputs = llm.generate(prompts, SamplingParams(max_tokens=100))

for output in outputs:
    # Get the first (and usually only) completion
    generated_text = output.outputs[0].text
    print(f"Input: {output.prompt}")
    print(f"Output: {generated_text}")
    print(f"Finish reason: {output.outputs[0].finish_reason}")
    print()
</syntaxhighlight>

=== Multiple Completions (n > 1) ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

# Generate 3 different completions
params = SamplingParams(n=3, temperature=0.8, max_tokens=100)
outputs = llm.generate(["Write a haiku about coding:"], params)

for output in outputs:
    print(f"Prompt: {output.prompt}\n")
    for i, completion in enumerate(output.outputs):
        print(f"Completion {i + 1}:")
        print(f"  Text: {completion.text}")
        print(f"  Cumulative logprob: {completion.cumulative_logprob}")
        print()
</syntaxhighlight>

=== Log Probability Analysis ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

# Request log probabilities
params = SamplingParams(
    temperature=0,
    max_tokens=20,
    logprobs=5,  # Top 5 logprobs per token
)

outputs = llm.generate(["The capital of France is"], params)

output = outputs[0]
completion = output.outputs[0]

print(f"Generated: {completion.text}")
print(f"\nToken-by-token probabilities:")
for i, (token_id, logprob_dict) in enumerate(zip(completion.token_ids, completion.logprobs)):
    print(f"  Token {i}: ID={token_id}, logprobs={logprob_dict}")
</syntaxhighlight>

=== Check Finish Reasons ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

params = SamplingParams(max_tokens=50, stop=[".", "!"])
outputs = llm.generate(["Write a sentence"], params)

for output in outputs:
    completion = output.outputs[0]

    if completion.finish_reason == "stop":
        print(f"Stopped due to: {completion.stop_reason}")
    elif completion.finish_reason == "length":
        print("Hit max_tokens limit")
    else:
        print("Generation incomplete")

    print(f"Text: {completion.text}")
</syntaxhighlight>

=== Prefix Cache Analysis ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

# Same prefix, different suffixes
prompts = [
    "The quick brown fox jumps over the lazy dog. Summarize:",
    "The quick brown fox jumps over the lazy dog. Translate to French:",
]

outputs = llm.generate(prompts, SamplingParams(max_tokens=50))

for output in outputs:
    cached = output.num_cached_tokens or 0
    total = len(output.prompt_token_ids) if output.prompt_token_ids else 0
    print(f"Cached tokens: {cached}/{total} ({100*cached/max(total,1):.1f}%)")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_Output_Processing]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
