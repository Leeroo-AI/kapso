# Principle: Transformer_Forward_Pass

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Attention Is All You Need|https://arxiv.org/abs/1706.03762]]
* [[source::Paper|Language Models are Unsupervised Multitask Learners|https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf]]
* [[source::Blog|The Illustrated GPT-2|https://jalammar.github.io/illustrated-gpt2/]]
* [[source::Repo|PicoGPT|https://github.com/jaymody/picoGPT]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::NLP]], [[domain::Transformers]], [[domain::Attention]]
|-
! Last Updated
| [[last_updated::2026-01-14 10:00 GMT]]
|}

== Overview ==

Computation that transforms a sequence of token IDs through stacked transformer blocks to produce output logits over the vocabulary.

=== Description ===

The Transformer Forward Pass is the core computation of GPT-2, taking token IDs and producing probability distributions over next tokens. The architecture consists of:

1. **Embedding Layer:** Token and positional embeddings are summed to create initial hidden states
2. **Transformer Blocks:** N identical layers, each containing:
   - Multi-head causal self-attention with residual connection
   - Position-wise feed-forward network with residual connection
   - Layer normalization (pre-norm variant in GPT-2)
3. **Output Projection:** Final hidden states are projected to vocabulary logits via weight tying with token embeddings

GPT-2 uses causal (autoregressive) attention masking, meaning each position can only attend to previous positions. This enables left-to-right text generation.

=== Usage ===

Use this principle when:
- Understanding how transformers process sequences
- Implementing inference for decoder-only language models
- Debugging or visualizing attention patterns
- Building educational implementations that expose transformer internals

The forward pass requires loaded model parameters and tokenized input; its output feeds into generation or loss computation.

== Theoretical Basis ==

**GPT-2 Architecture (decoder-only transformer):**

<syntaxhighlight lang="python">
# Abstract forward pass (shapes in comments)
def gpt2_forward(token_ids, params):
    # Embedding: [n_seq] -> [n_seq, n_embd]
    x = token_embeddings[token_ids] + position_embeddings[0:n_seq]

    # N transformer blocks: [n_seq, n_embd] -> [n_seq, n_embd]
    for block in transformer_blocks:
        x = x + multi_head_attention(layer_norm(x))  # Self-attention + residual
        x = x + feed_forward(layer_norm(x))          # FFN + residual

    # Output: [n_seq, n_embd] -> [n_seq, n_vocab]
    x = layer_norm(x)
    logits = x @ token_embeddings.T  # Weight tying
    return logits
</syntaxhighlight>

**Key Components:**

1. **Scaled Dot-Product Attention:**
<math>
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
</math>
Where M is the causal mask (−∞ for future positions, 0 for past).

2. **Multi-Head Attention:**
<math>
\text{MHA}(x) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
</math>
<math>
\text{head}_i = \text{Attention}(xW^Q_i, xW^K_i, xW^V_i)
</math>

3. **Feed-Forward Network:**
<math>
\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2
</math>
With hidden dimension = 4 × embedding dimension.

4. **GELU Activation:**
<math>
\text{GELU}(x) = 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right)
</math>

**GPT-2 Model Sizes:**

| Model | n_layer | n_head | n_embd | n_vocab | n_ctx |
|-------|---------|--------|--------|---------|-------|
| 124M  | 12      | 12     | 768    | 50257   | 1024  |
| 355M  | 24      | 16     | 1024   | 50257   | 1024  |
| 774M  | 36      | 20     | 1280   | 50257   | 1024  |
| 1558M | 48      | 25     | 1600   | 50257   | 1024  |

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Jaymody_PicoGPT_Gpt2]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:Jaymody_PicoGPT_Context_Length_Limits]]
