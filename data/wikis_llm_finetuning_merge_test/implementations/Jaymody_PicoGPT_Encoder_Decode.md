# Implementation: Encoder_Decode

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
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

Concrete tool for converting GPT-2 token IDs back to text strings provided by PicoGPT's Encoder class.

=== Description ===

The `Encoder.decode()` method converts a sequence of integer token IDs back into human-readable text. It performs the inverse of `encode()`:

1. Looks up each token ID in the decoder dictionary to get BPE subword strings
2. Concatenates all subwords
3. Maps BPE Unicode characters back to bytes
4. Decodes UTF-8 bytes to a Python string

The method handles invalid UTF-8 sequences according to the `errors` parameter set during Encoder initialization (default: "replace").

=== Usage ===

Use `Encoder.decode()` when:
- Converting generated token IDs to readable text
- Displaying model output to users
- Inspecting individual tokens

This is typically the final step in the generation pipeline.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/jaymody/picoGPT picoGPT]
* '''File:''' encoder.py
* '''Lines:''' 108-111

=== Signature ===
<syntaxhighlight lang="python">
def decode(self, tokens: list[int]) -> str:
    """
    Decode a list of token IDs to text.

    Args:
        tokens: list[int] - List of integer token IDs

    Returns:
        str - Decoded text string

    Note:
        - Invalid UTF-8 sequences handled by self.errors mode
        - Default error mode is "replace" (replaces invalid bytes with �)
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from encoder import get_encoder

encoder = get_encoder("124M", "models")
text = encoder.decode(token_ids)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| tokens || list[int] || Yes || List of integer token IDs to decode
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| text || str || Decoded text string
|}

== Usage Examples ==

=== Basic Decoding ===
<syntaxhighlight lang="python">
from encoder import get_encoder

encoder = get_encoder("124M", "models")

# Decode token IDs
tokens = [15496, 11, 995, 0]  # "Hello, world!"
text = encoder.decode(tokens)
print(text)  # "Hello, world!"
</syntaxhighlight>

=== Full Generation Pipeline ===
<syntaxhighlight lang="python">
from utils import load_encoder_hparams_and_params
from gpt2 import generate

# Load model
encoder, hparams, params = load_encoder_hparams_and_params("124M", "models")

# Encode prompt
prompt = "The future of AI is"
input_ids = encoder.encode(prompt)

# Generate new tokens
output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate=30)

# Decode ONLY the generated tokens
generated_text = encoder.decode(output_ids)
print(f"Prompt: {prompt}")
print(f"Generated: {generated_text}")
print(f"Full output: {prompt}{generated_text}")
</syntaxhighlight>

=== Inspecting Individual Tokens ===
<syntaxhighlight lang="python">
from encoder import get_encoder

encoder = get_encoder("124M", "models")

# Encode some text
text = "artificial intelligence"
tokens = encoder.encode(text)
print(f"Tokens: {tokens}")  # [433, 9596, 4430]

# Decode each token individually to see subwords
for i, tok in enumerate(tokens):
    subword = encoder.decode([tok])
    print(f"  Token {i}: {tok} -> '{subword}'")
# Token 0: 433 -> 'art'
# Token 1: 9596 -> 'ificial'
# Token 2: 4430 -> ' intelligence'  (note leading space)
</syntaxhighlight>

=== Handling Special Tokens ===
<syntaxhighlight lang="python">
from encoder import get_encoder

encoder = get_encoder("124M", "models")

# End of text token
eos_token = 50256
print(encoder.decode([eos_token]))  # "<|endoftext|>"

# Check if generation hit end token
generated = [15496, 11, 995, 50256, 262, 886]  # "Hello, world<|endoftext|> the end"
text = encoder.decode(generated)
print(text)  # "Hello, world<|endoftext|> the end"

# Often you want to stop at EOS
if 50256 in generated:
    eos_idx = generated.index(50256)
    text = encoder.decode(generated[:eos_idx])
    print(f"Truncated at EOS: '{text}'")  # "Hello, world"
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Jaymody_PicoGPT_Text_Decoding]]

=== Requires Environment ===
* [[requires_env::Environment:Jaymody_PicoGPT_Python_Dependencies]]
