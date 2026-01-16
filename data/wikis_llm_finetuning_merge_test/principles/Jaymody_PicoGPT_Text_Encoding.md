# Principle: Text_Encoding

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Neural Machine Translation of Rare Words with Subword Units|https://arxiv.org/abs/1508.07909]]
* [[source::Paper|Language Models are Unsupervised Multitask Learners|https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf]]
* [[source::Repo|PicoGPT|https://github.com/jaymody/picoGPT]]
|-
! Domains
| [[domain::NLP]], [[domain::Tokenization]], [[domain::Preprocessing]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==

Process of converting raw text input into a sequence of integer token IDs that can be processed by a language model.

=== Description ===

Text Encoding is the critical preprocessing step that transforms human-readable text into the numerical format expected by neural language models. For GPT-2, this involves:

1. **Text Splitting** - Use regex patterns to split text into "words" (handling contractions, punctuation, whitespace specially)
2. **Byte Encoding** - Convert each character to UTF-8 bytes, then map to Unicode characters
3. **BPE Application** - Apply learned byte-pair encoding merge rules to each word
4. **ID Lookup** - Convert BPE subword strings to integer token IDs

The encoding is deterministic - the same text always produces the same token sequence. This is essential for reproducibility and for matching the tokenization used during model training.

=== Usage ===

Use text encoding when:
- Preparing user input/prompts for GPT-2 inference
- Converting any text to tokens before feeding to the model
- Analyzing how specific text strings are tokenized

Text encoding is the first step of the inference pipeline after the model is loaded. The resulting token IDs are passed to the embedding layer of the transformer.

== Theoretical Basis ==

The encoding pipeline:

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# Text encoding pipeline
def encode(text: str) -> list[int]:
    token_ids = []

    # Step 1: Split text using regex pattern
    # Pattern handles: contractions, words, numbers, punctuation, whitespace
    words = regex.findall(pattern, text)

    for word in words:
        # Step 2: Convert to bytes and map to BPE vocabulary characters
        unicode_word = ""
        for byte in word.encode("utf-8"):
            unicode_word += byte_to_unicode[byte]

        # Step 3: Apply BPE merges to get subwords
        subwords = bpe(unicode_word)  # Returns "sub word s"

        # Step 4: Look up token IDs for each subword
        for subword in subwords.split(" "):
            token_ids.append(vocabulary[subword])

    return token_ids
</syntaxhighlight>

The regex pattern for GPT-2:
<syntaxhighlight lang="python">
pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
</syntaxhighlight>

This ensures contractions are handled correctly and whitespace is preserved appropriately.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Jaymody_PicoGPT_Encoder_Encode]]
