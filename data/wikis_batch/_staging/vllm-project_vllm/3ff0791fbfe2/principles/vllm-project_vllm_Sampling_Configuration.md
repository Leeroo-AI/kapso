# Sampling Configuration

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Nucleus Sampling|https://arxiv.org/abs/1904.09751]]
* [[source::Doc|OpenAI API Reference|https://platform.openai.com/docs/api-reference/completions]]
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Inference]], [[domain::Sampling]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:30 GMT]]
|}

== Overview ==

Configuration principle for controlling text generation behavior through temperature, top-p, top-k, and other sampling hyperparameters.

=== Description ===

Sampling Configuration governs how tokens are selected during autoregressive text generation. The sampling strategy determines the trade-off between output diversity and coherence.

Key sampling methods include:
* **Greedy decoding**: Select the highest probability token (temperature=0)
* **Temperature sampling**: Scale logits before softmax to control randomness
* **Top-k sampling**: Restrict to k highest probability tokens
* **Top-p (nucleus) sampling**: Restrict to smallest set with cumulative probability >= p
* **Min-p sampling**: Filter tokens below p * max_probability threshold

Proper sampling configuration is essential for balancing creativity and factual accuracy in generated text.

=== Usage ===

Configure sampling parameters when:
* Controlling output randomness (creative writing vs. factual Q&A)
* Setting generation length limits (max_tokens, min_tokens)
* Defining stop conditions (stop strings, stop tokens)
* Enabling log probability output for analysis
* Implementing structured output generation (JSON, regex constraints)

== Theoretical Basis ==

'''Temperature Scaling:'''
<math>
P(x_i) = \frac{exp(z_i / T)}{\sum_j exp(z_j / T)}
</math>

Where:
* <math>z_i</math> = logit for token i
* <math>T</math> = temperature (T→0 approaches greedy, T→∞ approaches uniform)

'''Top-p (Nucleus) Sampling:'''
<syntaxhighlight lang="python">
# Abstract algorithm for nucleus sampling
def nucleus_sample(probs, p):
    sorted_probs = sort_descending(probs)
    cumulative = cumsum(sorted_probs)
    cutoff_idx = first_index_where(cumulative >= p)
    nucleus = sorted_probs[:cutoff_idx + 1]
    return sample_from(normalize(nucleus))
</syntaxhighlight>

'''Repetition Penalties:'''
* **Presence penalty**: Penalize tokens that appear in output
* **Frequency penalty**: Penalize tokens proportional to their count
* **Repetition penalty**: Penalize tokens in both prompt and output

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_SamplingParams]]
