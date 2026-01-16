# Principle: Text_Decoding

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Neural Machine Translation of Rare Words with Subword Units|https://arxiv.org/abs/1508.07909]]
* [[source::Repo|PicoGPT|https://github.com/jaymody/picoGPT]]
* [[source::Repo|OpenAI GPT-2 Encoder|https://github.com/openai/gpt-2/blob/master/src/encoder.py]]
|-
! Domains
| [[domain::NLP]], [[domain::Tokenization]], [[domain::Text_Processing]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==

Process of converting a sequence of integer token IDs back into human-readable text.

=== Description ===

Text Decoding is the inverse of text encoding - it transforms model output (token IDs) back into a readable string. For GPT-2's BPE tokenizer, this involves:

1. **ID-to-Token Lookup** - Map each integer ID back to its BPE subword string
2. **Concatenation** - Join all subwords together
3. **Byte Decoding** - Convert Unicode characters back to UTF-8 bytes
4. **UTF-8 Interpretation** - Decode bytes to final text string

Unlike encoding, decoding is straightforward because the BPE subwords already represent the complete text when concatenated - no merge operations are needed.

=== Usage ===

Use text decoding when:
- Converting model output tokens to human-readable text
- Displaying generated text to users
- Debugging tokenization issues

Text decoding is the final step in the generation pipeline, applied to the token IDs produced by `generate()`.

== Theoretical Basis ==

The decoding process:

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# Text decoding pipeline
def decode(token_ids: list[int]) -> str:
    # Step 1: Look up each token ID in decoder dict
    subwords = [decoder[id] for id in token_ids]

    # Step 2: Concatenate subwords
    unicode_text = "".join(subwords)

    # Step 3: Convert BPE Unicode back to bytes
    bytes_list = [byte_decoder[char] for char in unicode_text]

    # Step 4: Decode UTF-8 bytes to string
    text = bytearray(bytes_list).decode("utf-8", errors=error_mode)

    return text
</syntaxhighlight>

Key properties:
- **Deterministic** - Same tokens always produce same text
- **Lossless** - encode(decode(tokens)) may differ but decode(encode(text)) == text
- **Error handling** - Invalid UTF-8 sequences handled by error mode ("replace", "ignore", "strict")

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Jaymody_PicoGPT_Encoder_Decode]]
