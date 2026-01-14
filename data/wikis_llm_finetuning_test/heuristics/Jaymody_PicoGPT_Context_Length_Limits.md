# Heuristic: Context_Length_Limits

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|picoGPT|https://github.com/jaymody/picoGPT]]
* [[source::Paper|GPT-2|https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf]]
|-
! Domains
| [[domain::NLP]], [[domain::Transformer]], [[domain::Constraints]]
|-
! Last Updated
| [[last_updated::2026-01-14 10:00 GMT]]
|}

== Overview ==
GPT-2 has a fixed context length of 1024 tokens; exceeding this causes assertion errors or truncation.

=== Description ===
The GPT-2 model's positional embeddings are learned for positions 0-1023, creating a hard limit of n_ctx=1024 tokens. PicoGPT enforces this limit with an assertion before generation. Input prompts plus desired generation length must stay under this limit. Longer contexts require chunking strategies or models with extended context windows.

=== Usage ===
Use this heuristic when:
- Planning input prompt length
- Setting `n_tokens_to_generate` parameter
- Encountering `AssertionError` during generation
- Working with long documents that need chunking

== The Insight (Rule of Thumb) ==
* **Action:** Always ensure `len(input_ids) + n_tokens_to_generate < n_ctx`
* **Value:** n_ctx = 1024 tokens for all GPT-2 model sizes
* **Trade-off:**
  - **Constraint:** Cannot process prompts longer than ~1000 tokens
  - **Workaround:** Chunk long inputs, summarize context, or use sliding window
* **Detection:** Check `hparams["n_ctx"]` for the exact limit

== Reasoning ==
The context length limit comes from the learned positional embeddings:
1. **wpe shape:** The positional embedding matrix has shape `[n_ctx, n_embd]`
2. **Position indexing:** Code uses `wpe[range(len(inputs))]` to lookup positions
3. **No extrapolation:** Positions beyond n_ctx have no learned embeddings

For longer contexts, consider:
- **Chunking:** Process text in overlapping windows
- **Summarization:** Compress earlier context into a summary
- **RoPE/ALiBi:** Use relative position encodings (not available in original GPT-2)

== Code Evidence ==

Context length assertion from `gpt2.py:106-107`:
<syntaxhighlight lang="python">
# make sure we are not surpassing the max sequence length of our model
assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]
</syntaxhighlight>

Positional embedding usage from `gpt2.py:75`:
<syntaxhighlight lang="python">
# token + positional embeddings
x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]
</syntaxhighlight>

Hyperparameters loaded from checkpoint include n_ctx from `utils.py:79`:
<syntaxhighlight lang="python">
hparams = json.load(open(os.path.join(model_dir, "hparams.json")))
</syntaxhighlight>

== Related Pages ==
* [[used_by::Implementation:Jaymody_PicoGPT_Generate]]
* [[used_by::Implementation:Jaymody_PicoGPT_Gpt2]]
* [[used_by::Principle:Jaymody_PicoGPT_Transformer_Forward_Pass]]
