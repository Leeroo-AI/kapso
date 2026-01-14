# Heuristic: No_KV_Cache_Performance

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|picoGPT|https://github.com/jaymody/picoGPT]]
* [[source::Blog|GPT in 60 Lines of Numpy|https://jaykmody.com/blog/gpt-from-scratch/]]
|-
! Domains
| [[domain::NLP]], [[domain::Optimization]], [[domain::Performance]]
|-
! Last Updated
| [[last_updated::2026-01-14 10:00 GMT]]
|}

== Overview ==
PicoGPT recomputes all attention keys/values on every token generation, causing O(n^2) complexity per step instead of O(n) with KV caching.

=== Description ===
Key-Value (KV) caching is a standard optimization in transformer inference that stores computed attention keys and values from previous tokens. PicoGPT intentionally omits this optimization for code clarity, meaning each new token requires a full forward pass over all previous tokens. This results in quadratic time complexity relative to sequence length.

=== Usage ===
Consider this heuristic when:
- Generation feels slow (it's expected behavior, not a bug)
- Generating long sequences (>100 tokens)
- Comparing PicoGPT performance to production implementations
- Deciding whether to add KV caching for a faster variant

== The Insight (Rule of Thumb) ==
* **Action:** PicoGPT runs full `gpt2()` forward pass for every generated token
* **Value:** No cached state between generation steps
* **Trade-off:**
  - **Pro:** Simpler code, easier to understand attention mechanism
  - **Con:** O(n^2 * L) total time for n tokens and L layers vs O(n * L) with cache
* **Impact:** Generating 40 tokens requires 40 full forward passes over the entire context

== Reasoning ==
The README explicitly states:
> "Fast? Nah, picoGPT is megaSLOW"

Without KV caching:
1. **Token 1:** Compute attention for positions [0]
2. **Token 2:** Recompute attention for positions [0, 1] (position 0 was already done!)
3. **Token n:** Recompute attention for positions [0, 1, ..., n-1]

With KV caching:
1. **Token 1:** Compute and cache K, V for position 0
2. **Token 2:** Compute K, V for position 1 only, reuse cached [0]
3. **Token n:** Compute K, V for position n-1 only, reuse cached [0..n-2]

For 40 tokens with context length 100:
- **Without cache:** 40 * (100 + 40) / 2 * 40 = ~112,000 attention operations
- **With cache:** 40 * 1 * 40 = 1,600 attention operations (70x fewer)

== Code Evidence ==

Full forward pass in generation loop from `gpt2.py:89-90`:
<syntaxhighlight lang="python">
for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
    logits = gpt2(inputs, **params, n_head=n_head)  # model forward pass
</syntaxhighlight>

Attention recomputation every forward pass from `gpt2.py:38-60`:
<syntaxhighlight lang="python">
def mha(x, c_attn, c_proj, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # qkv projection
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]
    # ... all Q, K, V recomputed from scratch
</syntaxhighlight>

Growing input list from `gpt2.py:92`:
<syntaxhighlight lang="python">
inputs.append(int(next_id))  # append prediction to input
</syntaxhighlight>

== Related Pages ==
* [[used_by::Implementation:Jaymody_PicoGPT_Generate]]
* [[used_by::Implementation:Jaymody_PicoGPT_Gpt2]]
* [[used_by::Principle:Jaymody_PicoGPT_Transformer_Forward_Pass]]
