# Heuristic: Greedy_Decoding_Tradeoffs

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|picoGPT|https://github.com/jaymody/picoGPT]]
* [[source::Blog|GPT in 60 Lines of Numpy|https://jaykmody.com/blog/gpt-from-scratch/]]
|-
! Domains
| [[domain::NLP]], [[domain::Text_Generation]], [[domain::Sampling]]
|-
! Last Updated
| [[last_updated::2026-01-14 10:00 GMT]]
|}

== Overview ==
Understanding the trade-offs of greedy decoding (argmax) vs stochastic sampling strategies for text generation quality and diversity.

=== Description ===
PicoGPT uses greedy decoding (argmax) for token selection, which always picks the highest probability token. This produces deterministic, consistent output but can lead to repetitive or "safe" text. Alternative sampling strategies like temperature scaling, top-k, top-p (nucleus), and categorical sampling introduce randomness for more diverse and creative outputs.

=== Usage ===
Use this heuristic when deciding on text generation quality vs diversity:
- **Greedy (argmax):** Best for factual, deterministic tasks where consistency matters
- **Temperature sampling:** When you need controlled randomness (higher temp = more creative)
- **Top-k/Top-p:** When you want diversity while avoiding low-probability "garbage" tokens

== The Insight (Rule of Thumb) ==
* **Action:** PicoGPT uses `np.argmax(logits[-1])` for greedy decoding
* **Value:** Always selects the single highest probability token (deterministic)
* **Trade-off:**
  - **Pro:** Simple, fast, reproducible results
  - **Con:** Can produce repetitive text, lacks creativity, may get stuck in loops
* **Alternatives not implemented:**
  - Temperature: `softmax(logits / T)` where T > 1 = more random, T < 1 = more focused
  - Top-k: Sample from top k most likely tokens only
  - Top-p (nucleus): Sample from smallest set of tokens exceeding cumulative probability p

== Reasoning ==
Greedy decoding is optimal for educational implementations because:
1. **Simplicity:** A single `np.argmax()` call vs complex sampling logic
2. **Reproducibility:** Same input always produces same output
3. **Debugging:** Easier to verify transformer correctness

The README explicitly notes this limitation:
> "top-p sampling? No. top-k? No. temperature? No. categorical sampling?! No. greedy? Yes."

For production text generation, consider implementing:
- **Temperature scaling** for controllable randomness
- **Top-k filtering** to avoid very low probability tokens
- **Top-p (nucleus) sampling** for dynamic vocabulary cutoff

== Code Evidence ==

Greedy sampling from `gpt2.py:91`:
<syntaxhighlight lang="python">
next_id = np.argmax(logits[-1])  # greedy sampling
</syntaxhighlight>

Full generation loop from `gpt2.py:86-94`:
<syntaxhighlight lang="python">
def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm

    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        logits = gpt2(inputs, **params, n_head=n_head)  # model forward pass
        next_id = np.argmax(logits[-1])  # greedy sampling
        inputs.append(int(next_id))  # append prediction to input

    return inputs[len(inputs) - n_tokens_to_generate :]  # only return generated ids
</syntaxhighlight>

== Related Pages ==
* [[used_by::Implementation:Jaymody_PicoGPT_Generate]]
* [[used_by::Principle:Jaymody_PicoGPT_Autoregressive_Generation]]
