# Implementation: Encoder_Encode

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

Concrete tool for converting text strings to token IDs using GPT-2's byte-level BPE tokenizer provided by the Encoder class.

=== Description ===

The `Encoder.encode` method tokenizes input text into a sequence of integer token IDs. The encoding process:

1. **Regex splitting:** Breaks text into chunks using a pattern that handles contractions, words, numbers, and punctuation
2. **Byte encoding:** Converts each chunk to UTF-8 bytes, then maps bytes to Unicode characters
3. **BPE merging:** Applies learned merge operations to compress each token
4. **Vocabulary lookup:** Maps BPE subwords to integer IDs

This implementation is copied from OpenAI's GPT-2 repository and handles arbitrary Unicode text without out-of-vocabulary issues.

=== Usage ===

Use this method to convert user prompts or input text into the token ID format expected by the GPT-2 forward pass. Must have an Encoder instance (from model loading) before calling.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/jaymody/picoGPT PicoGPT]
* '''File:''' encoder.py
* '''Lines:''' 101-106

=== Signature ===
<syntaxhighlight lang="python">
class Encoder:
    def encode(self, text: str) -> List[int]:
        """
        Encode a text string into a list of token IDs.

        Args:
            text: The input text to tokenize.

        Returns:
            List of integer token IDs.
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
| text || str || Yes || Raw text string to tokenize (any Unicode)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| (return) || List[int] || Sequence of token IDs indexing into vocabulary (0 to 50256)
|}

== Usage Examples ==

=== Basic Encoding ===
<syntaxhighlight lang="python">
from utils import load_encoder_hparams_and_params

# Load encoder
encoder, _, _ = load_encoder_hparams_and_params("124M", "models")

# Encode a simple string
text = "Hello, world!"
token_ids = encoder.encode(text)
print(token_ids)  # [15496, 11, 995, 0]

# Verify round-trip
decoded = encoder.decode(token_ids)
print(decoded)    # "Hello, world!"
assert decoded == text
</syntaxhighlight>

=== Inspecting Tokenization ===
<syntaxhighlight lang="python">
from utils import load_encoder_hparams_and_params

encoder, _, _ = load_encoder_hparams_and_params("124M", "models")

# See how text is split
text = "The quick brown fox jumps over the lazy dog."
token_ids = encoder.encode(text)

# Show individual tokens
for tid in token_ids:
    token = encoder.decoder[tid]
    print(f"{tid:5d} -> '{token}'")

# Output:
#   464 -> 'The'
#  2068 -> ' quick'
#  7586 -> ' brown'
#  ... etc
</syntaxhighlight>

=== Handling Special Cases ===
<syntaxhighlight lang="python">
from utils import load_encoder_hparams_and_params

encoder, _, _ = load_encoder_hparams_and_params("124M", "models")

# Unicode text (emojis, non-ASCII)
text = "Hello 世界! 🌍"
tokens = encoder.encode(text)
print(f"Tokens: {len(tokens)}")  # Unicode characters may use multiple tokens

# Contractions (handled by regex pattern)
text = "I'm can't won't"
tokens = encoder.encode(text)
decoded = encoder.decode(tokens)
assert decoded == text  # Lossless round-trip

# Whitespace preservation
text = "  multiple   spaces  "
tokens = encoder.encode(text)
decoded = encoder.decode(tokens)
assert decoded == text  # Whitespace is preserved
</syntaxhighlight>

=== Checking Sequence Length ===
<syntaxhighlight lang="python">
from utils import load_encoder_hparams_and_params

encoder, hparams, _ = load_encoder_hparams_and_params("124M", "models")

prompt = "Once upon a time in a land far away..."
n_tokens_to_generate = 50

# Tokenize and check length
input_ids = encoder.encode(prompt)
print(f"Prompt tokens: {len(input_ids)}")

# Ensure we don't exceed context length
max_seq_len = hparams["n_ctx"]  # 1024 for GPT-2
assert len(input_ids) + n_tokens_to_generate < max_seq_len, \
    f"Total length {len(input_ids) + n_tokens_to_generate} exceeds context {max_seq_len}"
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Jaymody_PicoGPT_Input_Tokenization]]

=== Requires Environment ===
* [[requires_env::Environment:Jaymody_PicoGPT_Python_Dependencies]]
