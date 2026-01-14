# Principle: Autoregressive_Generation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Language Models are Unsupervised Multitask Learners|https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf]]
* [[source::Blog|How to generate text with GPT-2|https://huggingface.co/blog/how-to-generate]]
* [[source::Repo|PicoGPT|https://github.com/jaymody/picoGPT]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::NLP]], [[domain::Text_Generation]], [[domain::Sampling]]
|-
! Last Updated
| [[last_updated::2026-01-14 10:00 GMT]]
|}

== Overview ==

Iterative decoding process that generates text one token at a time by repeatedly running the model forward pass and selecting the next token.

=== Description ===

Autoregressive Generation is the standard approach for producing text from language models. The process:

1. **Starts with a prompt** (sequence of token IDs)
2. **Runs the forward pass** to get logits for all positions
3. **Selects the next token** from the logits at the final position
4. **Appends the selected token** to the input sequence
5. **Repeats** until reaching the desired length or a stop condition

The "autoregressive" nature means each generated token becomes part of the input for generating the next token, creating a causal dependency chain. This differs from parallel decoding methods used in some encoder-decoder models.

=== Usage ===

Use this principle when:
- Generating text continuations from prompts
- Implementing chat or completion interfaces
- Understanding the inference-time behavior of language models
- Comparing different sampling strategies (greedy, top-k, nucleus)

Generation requires a loaded model, tokenized prompt, and typically decoding parameters (length, sampling method).

== Theoretical Basis ==

**Autoregressive Factorization:**

Language models decompose the joint probability of a sequence into conditional probabilities:

<math>
P(x_1, x_2, ..., x_n) = \prod_{i=1}^{n} P(x_i | x_1, ..., x_{i-1})
</math>

Each forward pass computes P(x_i | x_1, ..., x_{i-1}) for all positions simultaneously, but we only use the prediction at position i-1 when generating token i.

**Generation Loop (Pseudo-code):**

<syntaxhighlight lang="python">
def generate(prompt_ids, model, n_tokens):
    ids = prompt_ids.copy()
    for _ in range(n_tokens):
        logits = model.forward(ids)        # [n_seq, n_vocab]
        next_token = sample(logits[-1])    # Select from last position
        ids.append(next_token)
    return ids[len(prompt_ids):]           # Return only generated tokens
</syntaxhighlight>

**Sampling Strategies:**

| Strategy | Description | Use Case |
|----------|-------------|----------|
| Greedy | argmax(logits) | Deterministic, coherent text |
| Temperature | softmax(logits/T) | Control randomness (T<1: sharper, T>1: flatter) |
| Top-k | Sample from k highest logits | Balance diversity/quality |
| Nucleus (Top-p) | Sample from smallest set with cumulative prob ≥ p | Adaptive vocabulary |

**PicoGPT uses greedy decoding:**

<syntaxhighlight lang="python">
next_id = np.argmax(logits[-1])  # Always select highest probability token
</syntaxhighlight>

This produces deterministic output but may be repetitive or generic compared to sampling methods.

**Computational Cost:**

For sequence length L and generation length G:
- Naive: O(G × (L+G)²) — each step processes full sequence
- With KV cache: O(G × (L+G)) — reuse previous key/value computations

PicoGPT uses the naive approach for simplicity.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Jaymody_PicoGPT_Generate]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:Jaymody_PicoGPT_Greedy_Decoding_Tradeoffs]]
