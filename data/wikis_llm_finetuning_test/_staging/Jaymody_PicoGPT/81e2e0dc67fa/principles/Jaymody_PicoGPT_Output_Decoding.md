# Principle: Output_Decoding

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Neural Machine Translation of Rare Words with Subword Units|https://arxiv.org/abs/1508.07909]]
* [[source::Blog|Byte Pair Encoding|https://huggingface.co/learn/nlp-course/chapter6/5]]
* [[source::Repo|PicoGPT|https://github.com/jaymody/picoGPT]]
|-
! Domains
| [[domain::NLP]], [[domain::Tokenization]], [[domain::Text_Processing]]
|-
! Last Updated
| [[last_updated::2026-01-14 10:00 GMT]]
|}

== Overview ==

Process of converting model output token IDs back into human-readable text strings.

=== Description ===

Output Decoding is the inverse of tokenization, transforming sequences of integer token IDs back into text. For GPT-2's BPE tokenizer, this involves:

1. **Token ID to BPE token lookup:** Map each integer ID to its string representation in the vocabulary
2. **String concatenation:** Join all BPE tokens into a single string
3. **Byte decoding:** Convert the Unicode-encoded byte representation back to actual UTF-8 bytes
4. **UTF-8 decoding:** Convert bytes to a Python string

The decoding process must handle the byte-level encoding scheme that GPT-2 uses to ensure lossless round-trips for arbitrary Unicode text.

=== Usage ===

Use this principle when:
- Converting generated token IDs to displayable text
- Implementing completion/chat interfaces
- Debugging tokenization by examining round-trip behavior
- Understanding how subword tokenizers represent text

Decoding is the final step in text generation, occurring after autoregressive generation produces token IDs.

== Theoretical Basis ==

**Decoding Pipeline:**

<syntaxhighlight lang="python">
def decode(token_ids, vocab, byte_decoder):
    # Step 1: Token IDs -> BPE tokens
    bpe_tokens = [vocab[id] for id in token_ids]
    # e.g., [15496, 995] -> ["Hello", "Ġworld"]

    # Step 2: Concatenate
    text = "".join(bpe_tokens)
    # e.g., "HelloĠworld"

    # Step 3: Unicode -> Bytes
    bytes_list = [byte_decoder[char] for char in text]
    # e.g., [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]

    # Step 4: Bytes -> String
    result = bytes(bytes_list).decode("utf-8")
    # e.g., "Hello world"

    return result
</syntaxhighlight>

**Byte-to-Unicode Mapping:**

GPT-2 maps all 256 possible byte values to printable Unicode characters:

| Byte Range | Mapping |
|------------|---------|
| 33-126 (printable ASCII) | Identity (! to ~) |
| 161-172 (Latin-1 supplement) | Identity (¡ to ¬) |
| 174-255 (Latin-1 supplement) | Identity (® to ÿ) |
| All others (0-32, 127-160, 173) | Mapped to 256+ range |

This ensures:
- No whitespace/control characters in vocabulary tokens
- Lossless encoding of arbitrary byte sequences
- Human-readable vocabulary entries

**Error Handling:**

The decoder accepts an `errors` parameter (default: "replace") for handling invalid UTF-8 sequences:
- `"replace"`: Invalid bytes become U+FFFD (�)
- `"strict"`: Raises UnicodeDecodeError
- `"ignore"`: Invalid bytes are silently dropped

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Jaymody_PicoGPT_Encoder_Decode]]
