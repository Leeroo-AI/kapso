# Heuristic: Causal_Masking_Large_Negative

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|picoGPT|https://github.com/jaymody/picoGPT]]
* [[source::Paper|Attention Is All You Need|https://arxiv.org/abs/1706.03762]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::NLP]], [[domain::Transformers]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==
Use a large negative value (-1e10) instead of negative infinity for causal masking to ensure numerical stability in softmax.

=== Description ===
In autoregressive language models, future tokens must be masked during self-attention to prevent information leakage. The causal mask is added to the attention scores before softmax. While using `-inf` would mathematically produce zeros after softmax, it can cause numerical issues (NaN) in some implementations. Using a large finite negative like `-1e10` achieves the same effect while maintaining numerical stability.

=== Usage ===
Use this heuristic when implementing **Causal Self-Attention** or **Masked Multi-Head Attention** in transformer decoders. Apply this value to positions in the attention matrix that should not be attended to (future tokens in autoregressive models).

== The Insight (Rule of Thumb) ==
* **Action:** Create a causal mask using `(1 - np.tri(seq_len)) * -1e10`
* **Value:** `-1e10` as the mask value (not `-np.inf`)
* **Trade-off:** Negligible - the probability after softmax is effectively zero (exp(-1e10) approaches 0)
* **Compatibility:** Works with all numeric precisions (float32, float64); using `-np.inf` can cause NaN in some edge cases

== Reasoning ==
The softmax function computes `exp(x_i) / sum(exp(x_j))`. When masking with `-1e10`:
- `exp(-1e10) ≈ 0` (effectively zero for all practical purposes)
- The sum in the denominator remains finite and well-defined
- No risk of `inf - inf = NaN` computations

Using `-np.inf` can cause issues when:
- All positions are masked (rare edge case)
- Mixed precision computations occur
- Certain GPU implementations handle infinities differently

The value `-1e10` is chosen because:
- It's large enough that `exp(-1e10)` rounds to exactly 0 in float32/64
- It's small enough to not overflow when computing attention scores
- It's a common convention in ML frameworks

== Code Evidence ==

From `gpt2.py:L48-49` (multi-head attention):
<syntaxhighlight lang="python">
# causal mask to hide future inputs from being attended to
causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10  # [n_seq, n_seq]
</syntaxhighlight>

The mask is applied in `gpt2.py:L35` (attention function):
<syntaxhighlight lang="python">
def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v
</syntaxhighlight>

The `np.tri` function creates a lower triangular matrix of 1s, so `(1 - np.tri)` gives an upper triangular matrix with 1s above the diagonal. Multiplying by `-1e10` creates:
- 0 on and below the diagonal (positions that CAN be attended to)
- -1e10 above the diagonal (future positions that are MASKED)

== Related Pages ==
* [[used_by::Implementation:Jaymody_PicoGPT_Gpt2]]
* [[used_by::Principle:Jaymody_PicoGPT_Transformer_Architecture]]
