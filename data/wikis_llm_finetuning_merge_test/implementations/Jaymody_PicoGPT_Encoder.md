# Implementation: Encoder

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PicoGPT|https://github.com/jaymody/picoGPT]]
* [[source::Repo|OpenAI GPT-2 Encoder|https://github.com/openai/gpt-2/blob/master/src/encoder.py]]
|-
! Domains
| [[domain::NLP]], [[domain::Tokenization]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==

Concrete tool for byte-level BPE tokenization of text for GPT-2 models provided by PicoGPT (copied from OpenAI).

=== Description ===

The `Encoder` class implements GPT-2's byte-level BPE tokenizer. It converts text strings to sequences of integer token IDs and vice versa. The implementation is directly copied from OpenAI's official GPT-2 repository.

Key components:
- **encoder** - Dictionary mapping BPE tokens (strings) to integer IDs
- **decoder** - Reverse mapping from IDs to tokens
- **bpe_ranks** - Priority ordering of merge rules
- **byte_encoder/byte_decoder** - Mapping between UTF-8 bytes and Unicode characters

The tokenizer uses a regex pattern to first split text into "words" (including whitespace handling), then applies BPE to each word independently.

=== Usage ===

Use the Encoder class when:
- Tokenizing text input for GPT-2 inference
- Decoding model output tokens back to text
- Inspecting the tokenization of specific strings

The encoder is typically instantiated via `get_encoder()` helper function.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/jaymody/picoGPT picoGPT]
* '''File:''' encoder.py
* '''Lines:''' 47-111 (Encoder class), 114-120 (get_encoder)

=== Signature ===
<syntaxhighlight lang="python">
class Encoder:
    def __init__(
        self,
        encoder: dict,
        bpe_merges: list,
        errors: str = "replace"
    ) -> None:
        """
        Initialize the BPE encoder.

        Args:
            encoder: dict - Mapping from BPE tokens (str) to integer IDs
            bpe_merges: list - List of (str, str) tuples representing merge rules in priority order
            errors: str - Error handling mode for decoding ("replace", "strict", "ignore")
        """

    def bpe(self, token: str) -> str:
        """Apply BPE to a single token, returning space-separated subwords."""

    def encode(self, text: str) -> list[int]:
        """Encode text to list of token IDs."""

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs back to text."""


def get_encoder(model_name: str, models_dir: str) -> Encoder:
    """
    Load encoder from saved vocabulary files.

    Args:
        model_name: str - Model size ("124M", "355M", etc.)
        models_dir: str - Directory containing model files

    Returns:
        Encoder - Initialized encoder instance
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from encoder import Encoder, get_encoder
</syntaxhighlight>

== I/O Contract ==

=== Inputs (Encoder.__init__) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| encoder || dict || Yes || Token-to-ID mapping (50257 entries for GPT-2)
|-
| bpe_merges || list || Yes || List of merge rule tuples in priority order
|-
| errors || str || No || Decode error handling: "replace" (default), "strict", "ignore"
|}

=== Inputs (get_encoder) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model_name || str || Yes || Model variant: "124M", "355M", "774M", "1558M"
|-
| models_dir || str || Yes || Directory containing downloaded model files
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| encode() returns || list[int] || List of integer token IDs
|-
| decode() returns || str || Decoded text string
|-
| bpe() returns || str || Space-separated BPE subwords
|}

== Usage Examples ==

=== Basic Encoding/Decoding ===
<syntaxhighlight lang="python">
from encoder import get_encoder

# Load the encoder
encoder = get_encoder("124M", "models")

# Encode text to tokens
text = "Hello, world!"
tokens = encoder.encode(text)
print(tokens)  # [15496, 11, 995, 0]

# Decode tokens back to text
decoded = encoder.decode(tokens)
print(decoded)  # "Hello, world!"

# Verify round-trip
assert decoded == text
</syntaxhighlight>

=== Inspecting BPE Tokenization ===
<syntaxhighlight lang="python">
from encoder import get_encoder

encoder = get_encoder("124M", "models")

# See how a word gets split
text = "unbelievable"
tokens = encoder.encode(text)
print(f"'{text}' -> {tokens}")
# 'unbelievable' -> [403, 6667, 16665]

# Decode individual tokens to see subwords
for tok in tokens:
    print(f"  {tok} -> '{encoder.decode([tok])}'")
# 403 -> 'un'
# 6667 -> 'believ'
# 16665 -> 'able'
</syntaxhighlight>

=== Token Vocabulary Info ===
<syntaxhighlight lang="python">
from encoder import get_encoder

encoder = get_encoder("124M", "models")

# Vocabulary size
print(f"Vocab size: {len(encoder.encoder)}")  # 50257

# Check specific tokens
print(encoder.encoder.get("hello"))  # None (not a single token)
print(encoder.encoder.get("Hello"))  # None
print(encoder.encoder.get("Ġhello"))  # 31373 (Ġ = space prefix in GPT-2)

# Special tokens
print(encoder.encode("<|endoftext|>"))  # [50256]
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Jaymody_PicoGPT_BPE_Tokenization]]

=== Requires Environment ===
* [[requires_env::Environment:Jaymody_PicoGPT_Python_Dependencies]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:Jaymody_PicoGPT_BPE_Caching_LRU]]
