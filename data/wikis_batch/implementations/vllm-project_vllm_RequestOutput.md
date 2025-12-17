# RequestOutput

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::Inference]], [[domain::Output_Processing]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:30 GMT]]
|}

== Overview ==

Concrete tool for accessing generation results including text, tokens, log probabilities, and completion metadata from vLLM inference.

=== Description ===

`RequestOutput` is a dataclass containing all information about a completed generation request. It includes:

* The original prompt and its tokenized form
* One or more `CompletionOutput` objects (one per `n` value in SamplingParams)
* Optional log probabilities for prompt and generated tokens
* Request metrics and timing information
* Finish status and reason

The nested `CompletionOutput` class contains the actual generated content.

=== Usage ===

Access `RequestOutput` after calling `LLM.generate()` to:
* Get the generated text: `output.outputs[0].text`
* Access token IDs: `output.outputs[0].token_ids`
* Check finish reason: `output.outputs[0].finish_reason`
* Analyze log probabilities: `output.outputs[0].logprobs`

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm]
* '''File:''' vllm/outputs.py:L1-200

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class CompletionOutput:
    """The output data of one completion output of a request."""
    index: int
    """Index of this output (0 to n-1)."""

    text: str
    """Generated text."""

    token_ids: Sequence[int]
    """Token IDs of generated text."""

    cumulative_logprob: float | None
    """Total log probability of generation."""

    logprobs: SampleLogprobs | None
    """Per-token log probabilities (if requested)."""

    finish_reason: str | None = None
    """Why generation stopped: 'stop', 'length', etc."""

    stop_reason: int | str | None = None
    """The stop token/string that triggered stop."""


class RequestOutput:
    """The output data of a completion request to the LLM."""

    def __init__(
        self,
        request_id: str,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        prompt_logprobs: PromptLogprobs | None,
        outputs: list[CompletionOutput],
        finished: bool,
        metrics: RequestMetrics | RequestStateStats | None = None,
        lora_request: LoRARequest | None = None,
        encoder_prompt: str | None = None,
        encoder_prompt_token_ids: list[int] | None = None,
        num_cached_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm.outputs import RequestOutput, CompletionOutput
# Usually accessed from LLM.generate() return value
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| (from LLM.generate) || list[RequestOutput] || - || Returned by generate()
|}

=== Outputs (RequestOutput Attributes) ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| request_id || str || Unique identifier for this request
|-
| prompt || str || Input prompt text
|-
| prompt_token_ids || list[int] || Tokenized prompt
|-
| prompt_logprobs || list || Prompt token probabilities (if requested)
|-
| outputs || list[CompletionOutput] || Generated completions
|-
| finished || bool || Whether request is complete
|-
| metrics || RequestMetrics || Timing and performance data
|-
| num_cached_tokens || int || Tokens served from prefix cache
|}

=== Outputs (CompletionOutput Attributes) ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| index || int || Output index (0 to n-1)
|-
| text || str || Generated text
|-
| token_ids || list[int] || Generated token IDs
|-
| cumulative_logprob || float || Total log probability
|-
| logprobs || list[dict] || Per-token probabilities
|-
| finish_reason || str || "stop", "length", or None
|-
| stop_reason || str/int || What triggered stop
|}

== Usage Examples ==

=== Basic Output Access ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
prompts = ["What is AI?"]
outputs = llm.generate(prompts, SamplingParams(max_tokens=100))

# Access the generated text
result = outputs[0]
print(f"Request ID: {result.request_id}")
print(f"Prompt: {result.prompt}")
print(f"Generated: {result.outputs[0].text}")
print(f"Finish reason: {result.outputs[0].finish_reason}")
</syntaxhighlight>

=== Multiple Completions ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
prompts = ["Write a creative sentence:"]
params = SamplingParams(n=3, temperature=0.9, max_tokens=50)
outputs = llm.generate(prompts, params)

# Iterate over all completions
for i, completion in enumerate(outputs[0].outputs):
    print(f"Option {completion.index}: {completion.text}")
    print(f"  Log prob: {completion.cumulative_logprob}")
</syntaxhighlight>

=== With Log Probabilities ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
prompts = ["The capital of France is"]
params = SamplingParams(
    temperature=0,
    max_tokens=10,
    logprobs=5,           # Get top 5 logprobs per token
    prompt_logprobs=1,    # Also get prompt token logprobs
)

outputs = llm.generate(prompts, params)
result = outputs[0]

# Analyze prompt logprobs
if result.prompt_logprobs:
    print("Prompt token probabilities:")
    for i, token_logprob in enumerate(result.prompt_logprobs):
        if token_logprob:
            print(f"  Token {i}: {token_logprob}")

# Analyze generated token logprobs
completion = result.outputs[0]
if completion.logprobs:
    print("\nGenerated token probabilities:")
    for i, token_logprob in enumerate(completion.logprobs):
        print(f"  Position {i}: {token_logprob}")
</syntaxhighlight>

=== Token-Level Access ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
tokenizer = llm.get_tokenizer()

prompts = ["Hello, world!"]
outputs = llm.generate(prompts, SamplingParams(max_tokens=20))
result = outputs[0]

# Access token IDs
completion = result.outputs[0]
print(f"Generated tokens: {completion.token_ids}")

# Decode tokens individually
for token_id in completion.token_ids:
    token_str = tokenizer.decode([token_id])
    print(f"  {token_id} -> '{token_str}'")
</syntaxhighlight>

=== Check Finish Conditions ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
prompts = ["Count from 1 to 100:"]
params = SamplingParams(
    max_tokens=50,
    stop=["10", "twenty"],
)

outputs = llm.generate(prompts, params)
completion = outputs[0].outputs[0]

print(f"Generated: {completion.text}")
print(f"Finish reason: {completion.finish_reason}")
print(f"Stop reason: {completion.stop_reason}")

if completion.finish_reason == "stop":
    print("Generation stopped by stop sequence")
elif completion.finish_reason == "length":
    print("Generation stopped by max_tokens limit")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_Output_Processing]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_CUDA_Environment]]
