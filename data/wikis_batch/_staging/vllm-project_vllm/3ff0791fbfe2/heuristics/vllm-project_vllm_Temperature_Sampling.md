# Heuristic: vllm-project_vllm_Temperature_Sampling

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|OpenAI API|https://platform.openai.com/docs/api-reference/completions]]
|-
! Domains
| [[domain::LLMs]], [[domain::Sampling]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

Temperature parameter tuning to avoid numerical instability while controlling sampling randomness.

=== Description ===

The `temperature` parameter controls randomness in token sampling. Very low temperatures (< 0.01) can cause numerical issues (NaN/Inf in tensors), while temperature=0 triggers greedy decoding. vLLM automatically clamps temperatures below a minimum threshold to prevent these issues.

=== Usage ===

Use this heuristic when:
- Setting **temperature for generation** in SamplingParams
- Encountering **NaN or Inf errors** in logits
- Wanting **deterministic output** (use temperature=0 or seed)
- Balancing **creativity vs accuracy** in outputs

== The Insight (Rule of Thumb) ==

* **Action:** Set `temperature` in `SamplingParams`
* **Minimum Safe Value:** 0.01 (automatically clamped by vLLM)
* **Greedy Decoding:** temperature=0 triggers greedy sampling (most likely token)
* **Creative Output:** temperature=0.7-1.0 for diverse, creative responses
* **Factual Output:** temperature=0.1-0.3 for more deterministic, factual responses
* **Trade-off:** Higher temperature = more random; Lower = more deterministic

== Reasoning ==

Temperature scales the logits before softmax: `softmax(logits / temperature)`. Very small temperatures cause numerical overflow (division by near-zero). vLLM sets a minimum threshold (`_MAX_TEMP = 1e-2`) and logs a warning when clamping occurs. For true deterministic output, use `temperature=0` which bypasses temperature scaling and uses argmax directly.

== Code Evidence ==

From `vllm/sampling_params.py:21-22`:
<syntaxhighlight lang="python">
_SAMPLING_EPS = 1e-5
_MAX_TEMP = 1e-2
</syntaxhighlight>

Temperature clamping warning from `vllm/sampling_params.py:316-324`:
<syntaxhighlight lang="python">
def __post_init__(self) -> None:
    if 0 < self.temperature < _MAX_TEMP:
        logger.warning(
            "temperature %s is less than %s, which may cause numerical "
            "errors nan or inf in tensors. We have maxed it out to %s.",
            self.temperature,
            _MAX_TEMP,
            _MAX_TEMP,
        )
        self.temperature = max(self.temperature, _MAX_TEMP)
</syntaxhighlight>

Greedy sampling trigger from `vllm/sampling_params.py:353-358`:
<syntaxhighlight lang="python">
if self.temperature < _SAMPLING_EPS:
    # Zero temperature means greedy sampling.
    self.top_p = 1.0
    self.top_k = 0
    self.min_p = 0.0
    self._verify_greedy_sampling()
</syntaxhighlight>

== Related Pages ==

* [[uses_heuristic::Implementation:vllm-project_vllm_SamplingParams]]
* [[uses_heuristic::Workflow:vllm-project_vllm_Basic_Offline_LLM_Inference]]
* [[uses_heuristic::Workflow:vllm-project_vllm_Online_API_Serving]]
