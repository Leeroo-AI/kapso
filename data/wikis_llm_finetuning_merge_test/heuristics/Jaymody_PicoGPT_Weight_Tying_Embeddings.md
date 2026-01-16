# Heuristic: Weight_Tying_Embeddings

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|picoGPT|https://github.com/jaymody/picoGPT]]
* [[source::Paper|Using the Output Embedding to Improve Language Models|https://arxiv.org/abs/1608.05859]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::NLP]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==
Reuse the input token embedding matrix as the output projection layer (transposed) to reduce parameters and improve performance.

=== Description ===
Weight tying (also called weight sharing) is a technique where the input embedding matrix `wte` is reused as the output projection matrix for predicting the next token. Instead of learning separate matrices for embedding tokens and projecting hidden states to vocabulary logits, a single matrix is used (transposed for the output projection). This reduces parameters by `vocab_size * hidden_size` while often improving model quality.

=== Usage ===
Use this heuristic when implementing **Language Model Heads** for transformer-based text generation. After the final layer normalization, project the hidden states using `wte.T` (transposed token embeddings) to get vocabulary logits.

== The Insight (Rule of Thumb) ==
* **Action:** Use `x @ wte.T` for output projection instead of a separate learned matrix
* **Value:** Reuse `wte` (token embeddings) transposed
* **Trade-off:** Parameter reduction of ~50M parameters for GPT-2 small (50257 vocab x 768 hidden); slightly constrains model expressivity
* **Compatibility:** Standard practice in GPT-2, GPT-3, and many modern LLMs

== Reasoning ==
The intuition behind weight tying:
1. **Semantic consistency:** The embedding maps tokens to a semantic space; the output should project back from that same space
2. **Parameter efficiency:** For GPT-2 124M, the embedding matrix is 50257 x 768 = 38.6M parameters - tying saves learning these twice
3. **Regularization effect:** Forces the model to use consistent representations, acting as implicit regularization
4. **Empirical improvements:** Research shows weight tying often improves perplexity, especially on smaller datasets

Mathematical perspective:
- Input: `embedded = wte[token_ids]` - lookup rows of wte
- Output: `logits = hidden @ wte.T` - dot product with columns of wte (rows transposed)
- A token's logit is high when its hidden representation is similar to its embedding

== Code Evidence ==

From `gpt2.py:L73-83` (gpt2 forward pass):
<syntaxhighlight lang="python">
def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):  # [n_seq] -> [n_seq, n_vocab]
    # token + positional embeddings
    x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]

    # forward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, **ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]
</syntaxhighlight>

Note the key lines:
1. Line 75: `wte[inputs]` - token embedding lookup (input)
2. Line 83: `x @ wte.T` - same `wte` matrix transposed for output projection

The function signature shows `wte` is the only vocab-sized parameter passed; there is no separate `output_projection` weight.

== Related Pages ==
* [[used_by::Implementation:Jaymody_PicoGPT_Gpt2]]
* [[used_by::Principle:Jaymody_PicoGPT_Transformer_Architecture]]
