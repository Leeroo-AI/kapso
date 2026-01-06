{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|OpenAI API Reference|https://platform.openai.com/docs/api-reference/completions]]
|-
! Domains
| [[domain::NLP]], [[domain::Sampling]], [[domain::Text_Generation]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

Concrete tool for configuring text generation behavior including temperature, top-p/top-k sampling, repetition penalties, and stop conditions.

=== Description ===

`SamplingParams` is a configuration class that controls how the model generates tokens during inference. It follows the OpenAI API conventions for sampling parameters while adding vLLM-specific features like structured outputs and custom logits processors. The class is implemented using `msgspec` for efficient serialization.

=== Usage ===

Use `SamplingParams` whenever you call `LLM.generate()` to control:
- Generation randomness (temperature, top-p, top-k)
- Output length (max_tokens, min_tokens)
- Stop conditions (stop strings, stop token IDs)
- Penalty mechanisms (presence, frequency, repetition)
- Log probability output (logprobs, prompt_logprobs)

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project/vllm]
* '''File:''' vllm/sampling_params.py
* '''Lines:''' L111-241

=== Signature ===
<syntaxhighlight lang="python">
class SamplingParams(msgspec.Struct):
    """Sampling parameters for text generation."""

    n: int = 1
    """Number of output sequences to generate per prompt."""

    temperature: float = 1.0
    """Controls randomness. 0 = greedy, higher = more random."""

    top_p: float = 1.0
    """Nucleus sampling: cumulative probability threshold (0, 1]."""

    top_k: int = 0
    """Top-k sampling: number of tokens to consider. 0 = disabled."""

    min_p: float = 0.0
    """Minimum probability relative to most likely token [0, 1]."""

    seed: int | None = None
    """Random seed for reproducibility."""

    stop: str | list[str] | None = None
    """Stop strings that halt generation."""

    stop_token_ids: list[int] | None = None
    """Token IDs that halt generation."""

    max_tokens: int | None = 16
    """Maximum tokens to generate per sequence."""

    min_tokens: int = 0
    """Minimum tokens before stop conditions can trigger."""

    presence_penalty: float = 0.0
    """Penalty for tokens that appear in output (-2.0 to 2.0)."""

    frequency_penalty: float = 0.0
    """Penalty based on token frequency in output (-2.0 to 2.0)."""

    repetition_penalty: float = 1.0
    """Multiplicative penalty for repeated tokens (>1 discourages)."""

    logprobs: int | None = None
    """Number of top log probabilities to return per token."""

    prompt_logprobs: int | None = None
    """Number of log probabilities to return per prompt token."""

    structured_outputs: StructuredOutputsParams | None = None
    """Parameters for constrained/structured generation."""
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
| temperature || float || No || Sampling temperature (default: 1.0, 0 = greedy)
|-
| top_p || float || No || Nucleus sampling threshold (default: 1.0)
|-
| top_k || int || No || Top-k sampling (default: 0, disabled)
|-
| max_tokens || int || No || Maximum output length (default: 16)
|-
| stop || str | list[str] || No || Stop strings for generation
|-
| seed || int || No || Random seed for reproducibility
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| SamplingParams || SamplingParams || Configured sampling parameters instance
|}

== Usage Examples ==

=== Greedy Decoding ===
<syntaxhighlight lang="python">
from vllm import SamplingParams

# Deterministic output (greedy)
params = SamplingParams(
    temperature=0,
    max_tokens=100,
)
</syntaxhighlight>

=== Creative Generation ===
<syntaxhighlight lang="python">
from vllm import SamplingParams

# Higher temperature for creative tasks
params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    max_tokens=512,
    presence_penalty=0.1,   # Encourage diversity
    frequency_penalty=0.1,  # Reduce repetition
)
</syntaxhighlight>

=== Chat with Stop Strings ===
<syntaxhighlight lang="python">
from vllm import SamplingParams

# Stop at end of assistant turn
params = SamplingParams(
    temperature=0.7,
    max_tokens=256,
    stop=["</s>", "Human:", "\n\nHuman:"],
)
</syntaxhighlight>

=== Multiple Outputs with Logprobs ===
<syntaxhighlight lang="python">
from vllm import SamplingParams

# Generate multiple completions with probabilities
params = SamplingParams(
    n=3,                 # Generate 3 completions
    temperature=0.9,
    max_tokens=100,
    logprobs=5,          # Return top 5 logprobs per token
    seed=42,             # For reproducibility
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_Sampling_Parameters]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
