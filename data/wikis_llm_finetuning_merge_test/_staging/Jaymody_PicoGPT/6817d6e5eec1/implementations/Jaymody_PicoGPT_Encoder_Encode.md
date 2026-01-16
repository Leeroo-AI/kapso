# Implementation: Encoder_Encode

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PicoGPT|https://github.com/jaymody/picoGPT]]
* [[source::Repo|OpenAI GPT-2 Encoder|https://github.com/openai/gpt-2/blob/master/src/encoder.py]]
|-
! Domains
| [[domain::NLP]], [[domain::Tokenization]], [[domain::Preprocessing]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==

Concrete tool for converting text strings to GPT-2 token IDs provided by PicoGPT's Encoder class.

=== Description ===

The `Encoder.encode()` method is the primary interface for tokenizing text input. It takes a raw text string and returns a list of integer token IDs ready for model input.

The method internally:
1. Splits text using a regex pattern that handles English contractions and whitespace
2. Converts each word's characters to UTF-8 bytes mapped to special Unicode characters
3. Applies BPE via the `bpe()` method to split into subwords
4. Looks up each subword in the encoder dictionary to get token IDs

=== Usage ===

Use `Encoder.encode()` when:
- Preparing a prompt/input for GPT-2 text generation
- Converting user input to tokens before model inference
- Analyzing token counts or tokenization patterns

This is typically called once per input text, before passing tokens to `generate()`.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/jaymody/picoGPT picoGPT]
* '''File:''' encoder.py
* '''Lines:''' 101-106

=== Signature ===
<syntaxhighlight lang="python">
def encode(self, text: str) -> list[int]:
    """
    Encode a text string to a list of token IDs.

    Args:
        text: str - The input text to tokenize

    Returns:
        list[int] - List of integer token IDs

    Note:
        - The same text always produces the same tokens (deterministic)
        - Special handling for contractions ('s, 't, 're, etc.)
        - Whitespace is preserved as special tokens (Ġ prefix)
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from encoder import get_encoder

encoder = get_encoder("124M", "models")
tokens = encoder.encode("text")
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| text || str || Yes || Raw text string to tokenize (any UTF-8 text)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| tokens || list[int] || List of integer token IDs in range [0, 50256]
|}

== Usage Examples ==

=== Basic Text Encoding ===
<syntaxhighlight lang="python">
from encoder import get_encoder

encoder = get_encoder("124M", "models")

# Simple encoding
text = "Hello, world!"
tokens = encoder.encode(text)
print(tokens)  # [15496, 11, 995, 0]
print(f"Text length: {len(text)}, Token count: {len(tokens)}")
# Text length: 13, Token count: 4
</syntaxhighlight>

=== Encoding for GPT-2 Generation ===
<syntaxhighlight lang="python">
from encoder import get_encoder
from gpt2 import generate

encoder = get_encoder("124M", "models")

# Encode the prompt
prompt = "The meaning of life is"
input_ids = encoder.encode(prompt)
print(f"Input tokens: {input_ids}")  # [464, 3616, 286, 1204, 318]

# Check context length constraint
n_ctx = 1024  # GPT-2's context window
n_tokens_to_generate = 40
assert len(input_ids) + n_tokens_to_generate < n_ctx, "Input too long!"

# Now ready for generation
# output_ids = generate(input_ids, params, n_head, n_tokens_to_generate)
</syntaxhighlight>

=== Analyzing Tokenization ===
<syntaxhighlight lang="python">
from encoder import get_encoder

encoder = get_encoder("124M", "models")

# Different texts tokenize differently
texts = [
    "Hello",           # Common word -> few tokens
    "Pneumonoultramicroscopicsilicovolcanoconiosis",  # Long word -> many tokens
    "def main():",     # Code -> specific tokens
    "🎉",              # Emoji -> byte-level tokens
]

for text in texts:
    tokens = encoder.encode(text)
    print(f"'{text}' -> {len(tokens)} tokens: {tokens}")

# Output:
# 'Hello' -> 1 tokens: [15496]
# 'Pneumonoultramicroscopicsilicovolcanoconiosis' -> 15 tokens: [...]
# 'def main():' -> 4 tokens: [4299, 1388, 33529, 25]
# '🎉' -> 4 tokens: [8582, 236, 231, 223]
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Jaymody_PicoGPT_Text_Encoding]]

=== Requires Environment ===
* [[requires_env::Environment:Jaymody_PicoGPT_Python_Dependencies]]
