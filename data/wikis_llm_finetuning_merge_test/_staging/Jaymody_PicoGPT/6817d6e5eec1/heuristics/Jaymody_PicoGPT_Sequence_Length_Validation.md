# Heuristic: Sequence_Length_Validation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|picoGPT|https://github.com/jaymody/picoGPT]]
* [[source::Paper|Language Models are Unsupervised Multitask Learners|https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::NLP]], [[domain::Debugging]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==
Validate that input sequence length plus tokens to generate does not exceed the model's context window before generation.

=== Description ===
Transformer models have a fixed maximum context length (`n_ctx`) determined by the positional embedding size. GPT-2 has a context window of 1024 tokens. Exceeding this limit causes array indexing errors when looking up positional embeddings. Validating the total sequence length (input + generated tokens) upfront provides a clear error message and prevents cryptic failures during generation.

=== Usage ===
Use this heuristic when implementing **Text Generation** pipelines. Add an assertion or validation check before the generation loop to ensure `len(input_ids) + n_tokens_to_generate < n_ctx`.

== The Insight (Rule of Thumb) ==
* **Action:** Assert `len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]` before generation
* **Value:** GPT-2's `n_ctx` is 1024 tokens
* **Trade-off:** None - purely defensive programming that catches user errors early
* **Compatibility:** Applies to all fixed-context transformers (GPT-2, GPT-3, original Llama, etc.)

== Reasoning ==
Without validation, exceeding context length causes:
1. **Array out-of-bounds:** `wpe[range(len(inputs))]` fails when `len(inputs) > 1024`
2. **Cryptic error messages:** IndexError with no indication of the actual problem
3. **Silent truncation risk:** Some implementations silently truncate, leading to confusion

With validation:
1. **Clear error message:** User knows exactly what went wrong
2. **Actionable feedback:** User can shorten prompt or reduce `n_tokens_to_generate`
3. **Fail-fast principle:** Catch errors before expensive computation begins

Why `<` instead of `<=`:
- Strict inequality ensures at least one position for output
- Conservative approach avoids edge case issues
- In practice, users rarely need exactly 1024 tokens

== Code Evidence ==

From `gpt2.py:L106-107` (sequence length validation):
<syntaxhighlight lang="python">
# make sure we are not surpassing the max sequence length of our model
assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]
</syntaxhighlight>

This check occurs in the `main()` function before calling `generate()`:
<syntaxhighlight lang="python">
def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
</syntaxhighlight>

For GPT-2 124M, `hparams["n_ctx"]` is 1024, meaning:
- Prompt of 900 tokens + 123 generated tokens = 1023 total (OK)
- Prompt of 900 tokens + 124 generated tokens = 1024 total (FAILS assertion)

== Related Pages ==
* [[used_by::Implementation:Jaymody_PicoGPT_Generate]]
* [[used_by::Principle:Jaymody_PicoGPT_Autoregressive_Generation]]
