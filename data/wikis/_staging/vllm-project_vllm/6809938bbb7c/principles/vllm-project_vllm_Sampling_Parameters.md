{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|The Curious Case of Neural Text Degeneration|https://arxiv.org/abs/1904.09751]]
* [[source::Doc|OpenAI API Reference|https://platform.openai.com/docs/api-reference/completions]]
|-
! Domains
| [[domain::NLP]], [[domain::Sampling]], [[domain::Text_Generation]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

Configuration of stochastic decoding strategies that control the randomness, diversity, and quality of generated text.

=== Description ===

Sampling Parameters define how an LLM selects the next token during autoregressive generation. Unlike greedy decoding (always picking the most likely token), sampling-based methods introduce controlled randomness to produce more diverse and natural text.

Key sampling strategies include:
- **Temperature Scaling:** Adjusts the sharpness of the probability distribution
- **Top-K Sampling:** Restricts selection to K most likely tokens
- **Top-P (Nucleus) Sampling:** Selects from tokens comprising cumulative probability P
- **Penalty Mechanisms:** Discourage repetition of tokens or phrases

These parameters directly impact the quality, creativity, and coherence of generated outputs.

=== Usage ===

Configure sampling parameters when:
- Balancing creativity vs. accuracy in generation
- Reducing repetitive or degenerate outputs
- Ensuring reproducibility with fixed seeds
- Implementing chat applications with stop conditions
- Analyzing model confidence via log probabilities

Different tasks benefit from different configurations:
- **Factual Q&A:** Low temperature (0-0.3), greedy or near-greedy
- **Creative Writing:** Higher temperature (0.7-1.0), top-p sampling
- **Code Generation:** Medium temperature (0.2-0.5), careful stop strings
- **Chat/Dialogue:** Balanced settings with presence penalties

== Theoretical Basis ==

'''Temperature Scaling:'''

Given logits <math>z_i</math> for vocabulary token <math>i</math>, temperature <math>T</math> modifies the softmax:

<math>
P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
</math>

- <math>T \to 0</math>: Distribution becomes peaky (greedy)
- <math>T = 1</math>: Original distribution
- <math>T > 1</math>: Distribution flattens (more random)

'''Top-K Sampling:'''

Restrict sampling to top-K tokens by probability:
<math>
P'(x_i) = \begin{cases}
P(x_i) / \sum_{j \in K} P(x_j) & \text{if } i \in \text{top-K} \\
0 & \text{otherwise}
\end{cases}
</math>

'''Top-P (Nucleus) Sampling:'''

Select smallest set of tokens whose cumulative probability exceeds P:
<math>
V_p = \arg\min_{V' \subseteq V} \left| V' \right| \text{ s.t. } \sum_{x \in V'} P(x) \geq p
</math>

'''Repetition Penalties:'''

Presence penalty subtracts a constant for seen tokens:
<math>
z'_i = z_i - \alpha \cdot \mathbb{1}[i \in \text{seen}]
</math>

Frequency penalty scales with occurrence count:
<math>
z'_i = z_i - \beta \cdot \text{count}(i)
</math>

'''Pseudo-code:'''
<syntaxhighlight lang="python">
def sample_next_token(logits, params):
    # Apply temperature
    logits = logits / params.temperature

    # Apply penalties
    for token_id in generated_tokens:
        logits[token_id] -= params.presence_penalty
        logits[token_id] -= params.frequency_penalty * count[token_id]

    # Convert to probabilities
    probs = softmax(logits)

    # Apply top-k
    if params.top_k > 0:
        probs = top_k_filter(probs, params.top_k)

    # Apply top-p (nucleus)
    if params.top_p < 1.0:
        probs = nucleus_filter(probs, params.top_p)

    # Sample from filtered distribution
    return categorical_sample(probs, seed=params.seed)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_SamplingParams_init]]

=== Uses Heuristic ===
* [[uses_heuristic::Heuristic:vllm-project_vllm_Sampling_Temperature_Selection]]
