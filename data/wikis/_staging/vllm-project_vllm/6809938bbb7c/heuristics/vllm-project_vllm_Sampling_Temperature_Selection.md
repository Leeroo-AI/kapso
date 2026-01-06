# Sampling Temperature Selection

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|OpenAI API Sampling|https://platform.openai.com/docs/api-reference/completions/create]]
|-
! Domains
| [[domain::Sampling]], [[domain::Generation_Quality]], [[domain::LLM_Inference]]
|-
! Last Updated
| [[last_updated::2025-01-15 14:00 GMT]]
|}

== Overview ==

Guidance for selecting the `temperature` parameter to control output randomness, balancing between deterministic responses and creative variation.

=== Description ===

Temperature controls the randomness of token selection during generation. Lower temperatures make the model more deterministic (favoring high-probability tokens), while higher temperatures increase diversity and creativity (sampling from a flatter distribution). A temperature of 0 enables greedy decoding (always select the most likely token), while temperatures >1 make unlikely tokens more probable.

=== Usage ===

Use this heuristic to select appropriate temperature for:
- **Factual Q&A** (low temperature for consistency)
- **Creative writing** (higher temperature for diversity)
- **Code generation** (low temperature for correctness)
- **Brainstorming** (higher temperature for novel ideas)
- **Production deployments** (tune based on user feedback)

== The Insight (Rule of Thumb) ==

* **Action:** Set `temperature` in `SamplingParams`
* **Default Value:** 1.0
* **Valid Range:** 0.0 to 2.0 (typically; vLLM allows higher)
* **Trade-off:** Lower = more deterministic, Higher = more creative/random

{| class="wikitable"
! Temperature !! Behavior !! Best For
|-
| 0.0 || Greedy decoding (always most likely token) || Factual answers, code, deterministic outputs
|-
| 0.1 - 0.3 || Very focused, minor variation || Technical documentation, translations
|-
| 0.5 - 0.7 || Balanced creativity and coherence || General chat, customer service
|-
| 0.8 - 1.0 || Default randomness || Open-ended generation, storytelling
|-
| 1.2 - 1.5 || High creativity, some incoherence || Brainstorming, creative writing
|-
| > 1.5 || Very random, may be nonsensical || Experimental, artistic applications
|}

== Reasoning ==

'''Mathematical Effect:'''
Temperature divides the logits before softmax: `softmax(logits / temperature)`

- `temperature → 0`: softmax becomes one-hot (greedy)
- `temperature = 1`: standard softmax probabilities
- `temperature > 1`: flatter distribution, more uniform sampling

'''Practical Implications:'''
- **Repetition:** Lower temperatures can cause repetitive outputs; consider `repetition_penalty`
- **Coherence:** Higher temperatures may produce grammatically incorrect or nonsensical text
- **Consistency:** Temperature=0 guarantees same output for same input (deterministic)

'''Performance Note:'''
Temperature=0 enables greedy decoding which is slightly faster than sampling (no random number generation).

== Code Evidence ==

Temperature handling from `vllm/sampling_params.py:145-148`:
<syntaxhighlight lang="python">
temperature: float = 1.0
"""Controls the randomness of the sampling. Lower values make the model
more deterministic, while higher values make the model more random. Zero
means greedy sampling."""
</syntaxhighlight>

Sampling type selection based on temperature (`vllm/sampling_params.py:334-342`):
<syntaxhighlight lang="python">
@cached_property
def sampling_type(self) -> SamplingType:
    if self.temperature < _SAMPLING_EPS:
        return SamplingType.GREEDY
    if self.seed is not None:
        return SamplingType.RANDOM_SEED
    return SamplingType.RANDOM
</syntaxhighlight>

Constants from `vllm/sampling_params.py:21-22`:
<syntaxhighlight lang="python">
_SAMPLING_EPS = 1e-5  # Temperature below this triggers greedy
_MAX_TEMP = 1e-2
</syntaxhighlight>

== Usage Examples ==

=== Greedy Decoding (Deterministic) ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.2-1B")

# Deterministic output - same prompt always gives same response
params = SamplingParams(temperature=0)
output = llm.generate(["What is 2+2?"], params)
</syntaxhighlight>

=== Balanced Generation ===
<syntaxhighlight lang="python">
# Good balance for general conversation
params = SamplingParams(
    temperature=0.7,
    top_p=0.9,  # Combine with nucleus sampling
)
output = llm.generate(["Tell me about Python programming."], params)
</syntaxhighlight>

=== Creative Writing ===
<syntaxhighlight lang="python">
# Higher temperature for creative tasks
params = SamplingParams(
    temperature=1.2,
    top_p=0.95,
    max_tokens=500,
)
output = llm.generate(["Write a short poem about the ocean."], params)
</syntaxhighlight>

=== Code Generation ===
<syntaxhighlight lang="python">
# Low temperature for accurate code
params = SamplingParams(
    temperature=0.1,  # Low for correctness
    max_tokens=200,
)
output = llm.generate(["def fibonacci(n):"], params)
</syntaxhighlight>

== Combining with Other Parameters ==

Temperature works best when combined with other sampling parameters:

{| class="wikitable"
! Combination !! Effect !! Use Case
|-
| `temperature=0.7` + `top_p=0.9` || Nucleus sampling with moderate randomness || General-purpose generation
|-
| `temperature=0` + any top_p/top_k || Greedy (top_p/k ignored) || Deterministic outputs
|-
| `temperature=1.0` + `top_k=50` || Standard with vocabulary restriction || Controlled diversity
|-
| `temperature=0.5` + `repetition_penalty=1.1` || Focused but less repetitive || Long-form content
|}

== Common Pitfalls ==

{| class="wikitable"
! Pitfall !! Symptom !! Solution
|-
|| Temperature too low || Repetitive, "stuck" outputs || Increase to 0.5-0.7 or add `repetition_penalty`
|-
|| Temperature too high || Incoherent, nonsensical text || Reduce to 0.7-1.0
|-
|| Temperature=0 with n>1 || All n outputs are identical || Use temperature>0 for diverse outputs
|-
|| Ignoring top_p/top_k || Extreme temperatures less controllable || Combine with `top_p=0.9` for better control
|}

== Related Pages ==

* [[uses_heuristic::Implementation:vllm-project_vllm_SamplingParams_init]]
* [[uses_heuristic::Workflow:vllm-project_vllm_Basic_Offline_Inference]]
* [[uses_heuristic::Workflow:vllm-project_vllm_OpenAI_Compatible_Serving]]
* [[uses_heuristic::Principle:vllm-project_vllm_Sampling_Parameters]]
