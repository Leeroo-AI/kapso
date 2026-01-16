# Heuristic: Stable_Softmax

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|picoGPT|https://github.com/jaymody/picoGPT]]
* [[source::Blog|Softmax Numerical Stability|https://cs231n.github.io/linear-classify/#softmax]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Numerical_Stability]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==
Subtract the maximum value from logits before computing softmax to prevent numerical overflow.

=== Description ===
The softmax function computes `exp(x_i) / sum(exp(x_j))`. When input values are large, `exp(x)` can overflow to infinity, causing NaN results. Subtracting the maximum value from all inputs before exponentiation shifts the range so the largest value is 0 and all others are negative. Since `exp(0) = 1` and `exp(negative) < 1`, this prevents overflow while producing mathematically identical results (the subtraction cancels out in the ratio).

=== Usage ===
Use this heuristic whenever implementing **Softmax** from scratch. This is essential for attention mechanisms, classification heads, and any probability distribution computation over logits.

== The Insight (Rule of Thumb) ==
* **Action:** Compute `exp(x - max(x))` instead of `exp(x)` in softmax
* **Value:** Subtract `np.max(x, axis=-1, keepdims=True)` along the softmax dimension
* **Trade-off:** One extra `max()` operation (negligible cost)
* **Compatibility:** Universal - works in all numeric precisions and frameworks

== Reasoning ==
Mathematical proof that the result is unchanged:
```
softmax(x)_i = exp(x_i) / sum(exp(x_j))
            = exp(x_i - c) / sum(exp(x_j - c))  # multiply by exp(-c)/exp(-c) = 1
            = exp(x_i - c) / sum(exp(x_j - c))
```
Setting `c = max(x)` ensures:
- The largest exponent argument is 0: `exp(max(x) - max(x)) = exp(0) = 1`
- All other arguments are negative: `exp(x_i - max(x)) < 1` when `x_i < max(x)`
- No overflow possible since all values fed to `exp()` are <= 0

Without this trick:
- In float32, `exp(89)` ≈ 4.8e38 (near max float32 ~3.4e38)
- `exp(90)` = inf (overflow!)
- Once any value is inf, softmax outputs become NaN or invalid

This is one of the most important numerical stability tricks in deep learning.

== Code Evidence ==

From `gpt2.py:L8-10` (stable softmax implementation):
<syntaxhighlight lang="python">
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
</syntaxhighlight>

Key elements:
1. `np.max(x, axis=-1, keepdims=True)` - find max along last axis, keep dims for broadcasting
2. `x - np.max(...)` - shift all values so max is 0
3. `np.exp(...)` - now safe from overflow
4. Division by sum normalizes to probabilities

The `axis=-1` and `keepdims=True` ensure correct broadcasting when `x` is multi-dimensional (e.g., batched attention scores).

== Related Pages ==
* [[used_by::Implementation:Jaymody_PicoGPT_Gpt2]]
* [[used_by::Principle:Jaymody_PicoGPT_Transformer_Architecture]]
