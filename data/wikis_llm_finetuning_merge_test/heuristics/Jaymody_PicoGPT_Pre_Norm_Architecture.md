# Heuristic: Pre_Norm_Architecture

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|picoGPT|https://github.com/jaymody/picoGPT]]
* [[source::Paper|Language Models are Unsupervised Multitask Learners|https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::NLP]], [[domain::Transformers]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==
Apply layer normalization BEFORE attention and FFN blocks (pre-norm) rather than after (post-norm) for improved training stability.

=== Description ===
GPT-2 uses a "pre-norm" transformer architecture where layer normalization is applied at the input of each sub-layer (attention and FFN), before the transformation. This differs from the original "post-norm" Transformer from "Attention Is All You Need" which applied normalization after each sub-layer. Pre-norm architectures train more stably and often require less careful hyperparameter tuning.

=== Usage ===
Use this heuristic when implementing **Transformer Blocks** for language models. Apply layer normalization to the input before passing it to attention or FFN, then add the residual connection to the original input.

== The Insight (Rule of Thumb) ==
* **Action:** Apply `layer_norm` BEFORE `attention` and BEFORE `ffn`, not after
* **Value:** Residual structure: `x = x + sublayer(layer_norm(x))`
* **Trade-off:** Slightly different gradient flow compared to post-norm; empirically shown to be more stable for training large models
* **Compatibility:** This is the GPT-2/GPT-3 style; BERT uses post-norm

== Reasoning ==
Pre-norm places the layer normalization on the "residual branch" rather than the main path:
- **Post-norm (original Transformer):** `x = layer_norm(x + sublayer(x))`
- **Pre-norm (GPT-2 style):** `x = x + sublayer(layer_norm(x))`

Benefits of pre-norm:
1. **Gradient flow:** Gradients flow more directly through the residual connection without passing through normalization
2. **Training stability:** Reduces the need for learning rate warmup schedules
3. **Initialization sensitivity:** Less sensitive to weight initialization choices
4. **Scaling:** Better behaved as model depth increases

The intuition is that pre-norm ensures the residual connection carries unmodified information, while normalization stabilizes the transformation applied to that information.

== Code Evidence ==

From `gpt2.py:L63-70` (transformer block):
<syntaxhighlight lang="python">
def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # multi-head causal self attention
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, **ln_2), **mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x
</syntaxhighlight>

Note the structure:
1. `layer_norm(x, **ln_1)` is applied FIRST
2. Result passes through `mha()`
3. Original `x` is added via residual connection

This is clearly pre-norm: `x + mha(layer_norm(x))` not `layer_norm(x + mha(x))`.

== Related Pages ==
* [[used_by::Implementation:Jaymody_PicoGPT_Gpt2]]
* [[used_by::Principle:Jaymody_PicoGPT_Transformer_Architecture]]
