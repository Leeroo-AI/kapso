# Implementation: Encoder_Decode

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PicoGPT|https://github.com/jaymody/picoGPT]]
* [[source::Repo|OpenAI GPT-2|https://github.com/openai/gpt-2]]
|-
! Domains
| [[domain::NLP]], [[domain::Tokenization]], [[domain::Text_Processing]]
|-
! Last Updated
| [[last_updated::2026-01-14 10:00 GMT]]
|}

== Overview ==

Concrete tool for converting token IDs back to text strings using GPT-2's byte-level BPE decoder provided by the Encoder class.

=== Description ===

The `Encoder.decode` method converts sequences of integer token IDs back into human-readable text. The decoding process:

1. **Token lookup:** Maps each integer ID to its BPE string representation
2. **String concatenation:** Joins all BPE tokens into a single string
3. **Unicode to bytes:** Converts the Unicode-encoded characters back to byte values
4. **UTF-8 decoding:** Interprets bytes as a UTF-8 string with configurable error handling

This is the inverse of the `encode` method and enables lossless round-trips for arbitrary Unicode text.

=== Usage ===

Use this method to convert generated token IDs back to displayable text. Typically called as the final step in text generation after autoregressive decoding completes.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/jaymody/picoGPT PicoGPT]
* '''File:''' encoder.py
* '''Lines:''' 108-111

=== Signature ===
<syntaxhighlight lang="python">
class Encoder:
    def decode(self, tokens: List[int]) -> str:
        """
        Decode a list of token IDs back into a text string.

        Args:
            tokens: List of integer token IDs.

        Returns:
            Decoded text string.

        Note:
            Invalid UTF-8 byte sequences are handled according to
            self.errors (default: "replace" -> U+FFFD character).
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from encoder import Encoder, get_encoder

# Or via load_encoder_hparams_and_params:
from utils import load_encoder_hparams_and_params
encoder, hparams, params = load_encoder_hparams_and_params("124M", "models")
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| tokens || List[int] || Yes || Token IDs to decode (values 0-50256)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| (return) || str || Decoded text string
|}

== Usage Examples ==

=== Basic Decoding ===
<syntaxhighlight lang="python">
from utils import load_encoder_hparams_and_params

# Load encoder
encoder, _, _ = load_encoder_hparams_and_params("124M", "models")

# Decode token IDs
token_ids = [15496, 11, 995, 0]
text = encoder.decode(token_ids)
print(text)  # "Hello, world!"
</syntaxhighlight>

=== Decoding Generated Output ===
<syntaxhighlight lang="python">
from gpt2 import generate
from utils import load_encoder_hparams_and_params

encoder, hparams, params = load_encoder_hparams_and_params("124M", "models")

# Generate text
prompt = "The future of AI is"
input_ids = encoder.encode(prompt)
generated_ids = generate(input_ids, params, hparams["n_head"], 30)

# Decode generated tokens
generated_text = encoder.decode(generated_ids)
print(f"Generated: {generated_text}")

# Decode full sequence (prompt + generated)
full_text = encoder.decode(input_ids)  # input_ids was modified in-place
print(f"Full output: {full_text}")
</syntaxhighlight>

=== Round-Trip Verification ===
<syntaxhighlight lang="python">
from utils import load_encoder_hparams_and_params

encoder, _, _ = load_encoder_hparams_and_params("124M", "models")

# Test various inputs for lossless round-trip
test_strings = [
    "Hello, world!",
    "Multiple   spaces   preserved",
    "Unicode: café, naïve, 日本語",
    "Emojis: 🎉🚀💡",
    "Special chars: @#$%^&*()",
    "Contractions: I'm can't won't",
]

for original in test_strings:
    encoded = encoder.encode(original)
    decoded = encoder.decode(encoded)
    status = "✓" if decoded == original else "✗"
    print(f"{status} '{original}' -> {len(encoded)} tokens -> '{decoded}'")
</syntaxhighlight>

=== Inspecting Token Representations ===
<syntaxhighlight lang="python">
from utils import load_encoder_hparams_and_params

encoder, _, _ = load_encoder_hparams_and_params("124M", "models")

# Decode individual tokens to see their representation
sample_ids = [15496, 11, 262, 995]  # "Hello, the world"

for tid in sample_ids:
    # Raw BPE token (may contain Ġ for space)
    bpe_token = encoder.decoder[tid]

    # Decoded as text
    decoded = encoder.decode([tid])

    print(f"ID {tid:5d}: BPE='{bpe_token}' -> decoded='{decoded}'")

# Output:
# ID 15496: BPE='Hello' -> decoded='Hello'
# ID    11: BPE=',' -> decoded=','
# ID   262: BPE='Ġthe' -> decoded=' the'
# ID   995: BPE='Ġworld' -> decoded=' world'
</syntaxhighlight>

=== Error Handling Modes ===
<syntaxhighlight lang="python">
from encoder import Encoder, get_encoder
import os

# Default: errors="replace"
encoder = get_encoder("124M", "models")
# Invalid sequences become U+FFFD (replacement character)

# Create encoder with strict error handling
with open("models/124M/encoder.json") as f:
    import json
    vocab = json.load(f)
with open("models/124M/vocab.bpe", encoding="utf-8") as f:
    bpe_data = f.read()
bpe_merges = [tuple(merge.split()) for merge in bpe_data.split("\n")[1:-1]]

strict_encoder = Encoder(vocab, bpe_merges, errors="strict")
# Will raise UnicodeDecodeError on invalid sequences

ignore_encoder = Encoder(vocab, bpe_merges, errors="ignore")
# Invalid bytes are silently dropped
</syntaxhighlight>

=== Understanding the Decode Pipeline ===
<syntaxhighlight lang="python">
# The decode method implements:
#
# def decode(self, tokens):
#     # Step 1: Token ID -> BPE string
#     text = "".join([self.decoder[token] for token in tokens])
#
#     # Step 2: Unicode chars -> byte values
#     bytes_list = [self.byte_decoder[c] for c in text]
#
#     # Step 3: Bytes -> UTF-8 string
#     text = bytearray(bytes_list).decode("utf-8", errors=self.errors)
#
#     return text

# Example trace:
# [15496, 995] -> ["Hello", "Ġworld"] -> "HelloĠworld"
#              -> [72,101,108,108,111,32,119,111,114,108,100]
#              -> "Hello world"
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Jaymody_PicoGPT_Output_Decoding]]

=== Requires Environment ===
* [[requires_env::Environment:Jaymody_PicoGPT_Python_Dependencies]]
