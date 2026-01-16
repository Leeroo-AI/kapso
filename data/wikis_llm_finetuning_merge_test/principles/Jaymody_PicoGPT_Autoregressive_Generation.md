# Principle: Autoregressive_Generation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Language Models are Unsupervised Multitask Learners|https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf]]
* [[source::Paper|Attention Is All You Need|https://arxiv.org/abs/1706.03762]]
* [[source::Blog|The Illustrated GPT-2|https://jalammar.github.io/illustrated-gpt2/]]
* [[source::Repo|PicoGPT|https://github.com/jaymody/picoGPT]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::NLP]], [[domain::Text_Generation]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==

Sequential text generation method where each new token is predicted based on all previously generated tokens.

=== Description ===

Autoregressive Generation is the standard approach for generating text with language models like GPT-2. The model generates one token at a time, where each new token depends on:
1. The original input prompt tokens
2. All tokens generated so far

This creates a sequential dependency chain where position <math>t</math> can only attend to positions <math>< t</math>. The process continues until either:
- A specified number of tokens have been generated
- An end-of-sequence token is produced
- A maximum context length is reached

The "autoregressive" name comes from the fact that the model's output becomes part of its input for subsequent predictions.

=== Usage ===

Use autoregressive generation when:
- Generating text completions from prompts
- Implementing language model inference
- Building chatbots, story generators, or code completion systems

This is the core generation loop for all GPT-style models. Alternative methods (like parallel decoding or speculative decoding) exist but are more complex.

== Theoretical Basis ==

Autoregressive generation factorizes the joint probability of a sequence:

<math>
P(x_1, x_2, ..., x_T) = \prod_{t=1}^{T} P(x_t | x_1, ..., x_{t-1})
</math>

At each step, the model outputs a probability distribution over the vocabulary:

<math>
P(x_t | x_{<t}) = \text{softmax}(\text{Transformer}(x_1, ..., x_{t-1}))
</math>

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# Autoregressive generation loop
def generate(prompt_tokens: list[int], n_tokens: int) -> list[int]:
    tokens = prompt_tokens.copy()

    for _ in range(n_tokens):
        # Forward pass through transformer
        logits = model_forward(tokens)  # [seq_len, vocab_size]

        # Get logits for last position only
        next_token_logits = logits[-1]  # [vocab_size]

        # Select next token (greedy: argmax)
        next_token = argmax(next_token_logits)

        # Append to sequence
        tokens.append(next_token)

    return tokens[len(prompt_tokens):]  # Return only generated tokens
</syntaxhighlight>

'''Sampling strategies:'''
- '''Greedy''' (used in PicoGPT): Always pick highest probability token
- '''Top-k''': Sample from top k highest probability tokens
- '''Top-p (nucleus)''': Sample from smallest set with cumulative probability > p
- '''Temperature''': Scale logits before softmax to control randomness

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Jaymody_PicoGPT_Generate]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:Jaymody_PicoGPT_Sequence_Length_Validation]]
