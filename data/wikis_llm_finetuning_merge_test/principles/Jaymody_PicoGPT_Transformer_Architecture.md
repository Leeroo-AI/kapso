# Principle: Transformer_Architecture

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Attention Is All You Need|https://arxiv.org/abs/1706.03762]]
* [[source::Paper|Language Models are Unsupervised Multitask Learners|https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf]]
* [[source::Blog|The Illustrated Transformer|https://jalammar.github.io/illustrated-transformer/]]
* [[source::Blog|The Illustrated GPT-2|https://jalammar.github.io/illustrated-gpt2/]]
* [[source::Repo|PicoGPT|https://github.com/jaymody/picoGPT]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::NLP]], [[domain::Model_Architecture]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==

Neural network architecture that uses self-attention mechanisms to process sequential data in parallel without recurrence.

=== Description ===

The Transformer architecture, introduced in "Attention Is All You Need" (2017), revolutionized sequence modeling by replacing recurrent mechanisms with self-attention. GPT-2 uses a decoder-only Transformer variant optimized for autoregressive language modeling.

Key components of the GPT-2 architecture:

1. **Token Embeddings (wte)** - Maps token IDs to dense vectors [n_vocab, n_embd]
2. **Positional Embeddings (wpe)** - Learned position representations [n_ctx, n_embd]
3. **Transformer Blocks** - Stack of N identical layers, each containing:
   - Layer Normalization (pre-norm style)
   - Multi-Head Causal Self-Attention
   - Another Layer Normalization
   - Feed-Forward Network (MLP)
   - Residual connections around both
4. **Final Layer Norm (ln_f)** - Applied before output projection
5. **Output Projection** - Shared with token embeddings (weight tying)

GPT-2 uses **pre-norm** architecture where layer norm is applied before attention/MLP, rather than after (post-norm). This improves training stability for deep models.

=== Usage ===

Use the Transformer architecture when:
- Building language models for text generation
- Implementing encoder-decoder or decoder-only models
- Working with sequence data where long-range dependencies matter

GPT-2 variants differ only in size parameters:
| Model | n_layer | n_head | n_embd |
|-------|---------|--------|--------|
| 124M  | 12      | 12     | 768    |
| 355M  | 24      | 16     | 1024   |
| 774M  | 36      | 20     | 1280   |
| 1558M | 48      | 25     | 1600   |

== Theoretical Basis ==

The GPT-2 forward pass computes:

<math>
h_0 = W_e[x] + W_p[0:n]
</math>

<math>
h_l = \text{TransformerBlock}(h_{l-1}) \quad \text{for } l = 1...L
</math>

<math>
\text{logits} = \text{LayerNorm}(h_L) \cdot W_e^T
</math>

Each Transformer Block:
<math>
h' = h + \text{MHA}(\text{LN}(h))
</math>
<math>
h'' = h' + \text{FFN}(\text{LN}(h'))
</math>

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# GPT-2 Forward Pass
def gpt2_forward(token_ids: list[int]) -> np.ndarray:
    # 1. Embeddings
    x = token_embeddings[token_ids] + position_embeddings[0:len(token_ids)]

    # 2. Transformer Blocks (pre-norm style)
    for block in blocks:
        # Multi-head attention with residual
        x = x + mha(layer_norm(x, block.ln_1), block.attn)

        # Feed-forward network with residual
        x = x + ffn(layer_norm(x, block.ln_2), block.mlp)

    # 3. Final layer norm + projection to vocabulary
    x = layer_norm(x, ln_f)
    logits = x @ token_embeddings.T  # Weight tying

    return logits  # [seq_len, vocab_size]
</syntaxhighlight>

Key properties:
- **Causal masking**: Position i can only attend to positions < i
- **Pre-norm**: LayerNorm before attention/FFN, not after
- **Weight tying**: Output projection reuses token embeddings transposed
- **GELU activation**: In FFN, uses GELU instead of ReLU

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Jaymody_PicoGPT_Gpt2]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:Jaymody_PicoGPT_Causal_Masking_Large_Negative]]
* [[uses_heuristic::Heuristic:Jaymody_PicoGPT_Pre_Norm_Architecture]]
* [[uses_heuristic::Heuristic:Jaymody_PicoGPT_Weight_Tying_Embeddings]]
* [[uses_heuristic::Heuristic:Jaymody_PicoGPT_Stable_Softmax]]
