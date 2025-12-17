# SamplingParams

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
* [[source::Doc|OpenAI API Reference|https://platform.openai.com/docs/api-reference/completions]]
|-
! Domains
| [[domain::Inference]], [[domain::Sampling]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:30 GMT]]
|}

== Overview ==

Concrete tool for configuring text generation sampling parameters including temperature, top-p, stop sequences, and output length limits.

=== Description ===

`SamplingParams` is a msgspec struct that defines how tokens are selected during text generation. It follows the OpenAI API conventions while adding vLLM-specific optimizations like beam search support and structured output constraints.

The class uses msgspec for efficient serialization and pydantic integration for validation, making it suitable for both high-throughput batch processing and API request handling.

=== Usage ===

Use `SamplingParams` when calling `LLM.generate()` to control:
* Output randomness and diversity
* Generation length and stop conditions
* Log probability output
* Structured output constraints (JSON schema, regex)

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm]
* '''File:''' vllm/sampling_params.py:L111-597

=== Signature ===
<syntaxhighlight lang="python">
class SamplingParams(
    PydanticMsgspecMixin,
    msgspec.Struct,
    omit_defaults=True,
    dict=True,
):
    """Sampling parameters for text generation."""

    n: int = 1
    """Number of outputs per prompt."""

    temperature: float = 1.0
    """Sampling temperature. 0 = greedy."""

    top_p: float = 1.0
    """Nucleus sampling probability threshold."""

    top_k: int = 0
    """Top-k sampling. 0 = disabled."""

    min_p: float = 0.0
    """Min-p sampling threshold."""

    max_tokens: int | None = 16
    """Maximum tokens to generate."""

    stop: str | list[str] | None = None
    """Stop string(s) for generation."""

    stop_token_ids: list[int] | None = None
    """Token IDs that stop generation."""

    presence_penalty: float = 0.0
    """Penalty for token presence."""

    frequency_penalty: float = 0.0
    """Penalty proportional to frequency."""

    repetition_penalty: float = 1.0
    """Penalty for prompt/output tokens."""

    seed: int | None = None
    """Random seed for reproducibility."""

    logprobs: int | None = None
    """Number of logprobs to return per token."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm import SamplingParams
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| temperature || float || No || Controls randomness (0-2, default: 1.0)
|-
| top_p || float || No || Nucleus sampling threshold (0-1, default: 1.0)
|-
| top_k || int || No || Top-k filtering (0=disabled, default: 0)
|-
| max_tokens || int || No || Max tokens to generate (default: 16)
|-
| stop || str/list || No || Stop string(s) for generation
|-
| seed || int || No || Random seed for reproducibility
|-
| logprobs || int || No || Number of logprobs per token (None=disabled)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| SamplingParams || msgspec.Struct || Immutable sampling configuration object
|}

== Usage Examples ==

=== Greedy Decoding ===
<syntaxhighlight lang="python">
from vllm import SamplingParams

# Deterministic output (best for factual tasks)
params = SamplingParams(
    temperature=0,
    max_tokens=512,
)
</syntaxhighlight>

=== Creative Generation ===
<syntaxhighlight lang="python">
from vllm import SamplingParams

# High diversity output (best for creative writing)
params = SamplingParams(
    temperature=0.9,
    top_p=0.95,
    max_tokens=1024,
    presence_penalty=0.6,  # Encourage novel tokens
)
</syntaxhighlight>

=== Chat Completion with Stop Tokens ===
<syntaxhighlight lang="python">
from vllm import SamplingParams

# Chat-style generation with stop sequences
params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
    stop=["</s>", "Human:", "\n\n"],
)
</syntaxhighlight>

=== Structured Output (JSON) ===
<syntaxhighlight lang="python">
from vllm import SamplingParams

# Constrain output to JSON schema
params = SamplingParams(
    temperature=0.7,
    max_tokens=256,
    structured_output={
        "type": "json_object",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
    }
)
</syntaxhighlight>

=== With Log Probabilities ===
<syntaxhighlight lang="python">
from vllm import SamplingParams

# Get top-5 log probabilities for analysis
params = SamplingParams(
    temperature=0.7,
    max_tokens=100,
    logprobs=5,
    prompt_logprobs=1,  # Also get prompt token logprobs
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_Sampling_Configuration]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_CUDA_Environment]]
