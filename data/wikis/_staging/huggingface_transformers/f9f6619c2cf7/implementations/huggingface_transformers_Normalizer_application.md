{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Tokenizers Normalizers|https://huggingface.co/docs/tokenizers/api/normalizers]]
* [[source::Repo|HuggingFace Tokenizers|https://github.com/huggingface/tokenizers]]
|-
! Domains
| [[domain::NLP]], [[domain::Preprocessing]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Concrete tool for applying text normalization using the HuggingFace tokenizers library's Normalizer components.

=== Description ===
This implementation wraps the tokenizers library's normalizer functionality to provide text normalization as part of the Transformers tokenization pipeline. The tokenizers library provides efficient Rust-based normalizer components that can be composed into sequences. While Transformers doesn't directly expose normalizer configuration in most cases (it's embedded in the tokenizer.json file for fast tokenizers), understanding how normalizers work helps when customizing tokenization or debugging preprocessing issues.

Fast tokenizers (PreTrainedTokenizerFast) automatically apply configured normalizers when tokenizing. The normalizer configuration is saved in tokenizer.json and loaded when the tokenizer is initialized. For custom tokenization needs, normalizers can be configured using the tokenizers library API.

=== Usage ===
Use this implementation when:
* Working with fast tokenizers that have embedded normalizer configuration
* Building custom tokenizers with specific normalization requirements
* Debugging normalization behavior in existing tokenizers
* Creating domain-specific tokenizers with custom preprocessing
* Understanding how pretrained tokenizers preprocess text

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/tokenizers tokenizers]
* '''File:''' tokenizers library (Rust implementation with Python bindings)
* '''Transformers Integration:''' Fast tokenizers automatically apply normalizers configured in tokenizer.json

=== Signature ===
<syntaxhighlight lang="python">
# From tokenizers library
from tokenizers import normalizers

# Create individual normalizers
nfd_normalizer = normalizers.NFD()
lowercase_normalizer = normalizers.Lowercase()
strip_accents_normalizer = normalizers.StripAccents()

# Compose normalizers into sequence
normalizer = normalizers.Sequence([
    normalizers.NFD(),
    normalizers.Lowercase(),
    normalizers.StripAccents()
])

# Apply normalization
normalized_text = normalizer.normalize_str(text)
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# For using normalizers directly
from tokenizers import normalizers

# Fast tokenizers apply normalizers automatically
from transformers import AutoTokenizer
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| text || str || Yes || Raw input text to normalize
|-
| normalizer || normalizers.Normalizer || Yes || Configured normalizer or sequence of normalizers to apply
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| normalized_text || str || Text after applying all normalization transformations
|}

== Usage Examples ==

=== Example: Automatic Normalization with Fast Tokenizers ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

# Load fast tokenizer (normalizer is embedded in tokenizer.json)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Normalization happens automatically during tokenization
text = "Héllo WORLD!!! Café"

# Tokenize (normalization applied internally)
tokens = tokenizer.tokenize(text)
print(tokens)
# Output: ['hello', 'world', '!', '!', '!', 'cafe']
# Note: Accents removed, text lowercased

# The normalizer is accessible
if hasattr(tokenizer, 'backend_tokenizer'):
    normalizer = tokenizer.backend_tokenizer.normalizer
    normalized = normalizer.normalize_str(text)
    print(f"Normalized: '{normalized}'")
    # Output: 'hello world!!! cafe'
</syntaxhighlight>

=== Example: Building Custom Normalizer ===
<syntaxhighlight lang="python">
from tokenizers import normalizers, Tokenizer
from tokenizers.models import WordPiece

# Create custom normalizer sequence
normalizer = normalizers.Sequence([
    normalizers.NFD(),         # Decompose Unicode
    normalizers.Lowercase(),   # Convert to lowercase
    normalizers.StripAccents() # Remove accent marks
])

# Apply normalizer directly
text = "Café Münchën"
normalized = normalizer.normalize_str(text)
print(f"Original: {text}")
print(f"Normalized: {normalized}")
# Output:
# Original: Café Münchën
# Normalized: cafe munchen

# Create tokenizer with custom normalizer
tokenizer = Tokenizer(WordPiece())
tokenizer.normalizer = normalizer
</syntaxhighlight>

=== Example: Different Normalization Strategies ===
<syntaxhighlight lang="python">
from tokenizers import normalizers

text = "Héllo WORLD!!!   Multiple   spaces"

# Strategy 1: BERT-style (NFD + Lowercase + StripAccents)
bert_normalizer = normalizers.Sequence([
    normalizers.NFD(),
    normalizers.Lowercase(),
    normalizers.StripAccents()
])
print(f"BERT-style: '{bert_normalizer.normalize_str(text)}'")
# Output: 'hello world!!!   multiple   spaces'

# Strategy 2: NFC (compose to canonical form)
nfc_normalizer = normalizers.NFC()
print(f"NFC: '{nfc_normalizer.normalize_str(text)}'")
# Output: 'Héllo WORLD!!!   Multiple   spaces' (composed Unicode)

# Strategy 3: Lowercase only
lowercase_normalizer = normalizers.Lowercase()
print(f"Lowercase: '{lowercase_normalizer.normalize_str(text)}'")
# Output: 'héllo world!!!   multiple   spaces'

# Strategy 4: No normalization
print(f"No normalization: '{text}'")
# Output: 'Héllo WORLD!!!   Multiple   spaces'
</syntaxhighlight>

=== Example: Inspecting Tokenizer Normalization ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

# Compare normalization across different models
models = ["bert-base-uncased", "gpt2", "roberta-base"]
text = "Héllo WORLD!!!"

for model_name in models:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Check if it's a fast tokenizer with normalizer
    if hasattr(tokenizer, 'backend_tokenizer') and tokenizer.backend_tokenizer.normalizer:
        normalizer = tokenizer.backend_tokenizer.normalizer
        normalized = normalizer.normalize_str(text)
        print(f"{model_name}: '{normalized}'")
    else:
        print(f"{model_name}: No accessible normalizer (slow tokenizer)")

# Output might show:
# bert-base-uncased: 'hello world!!!'
# gpt2: 'Héllo WORLD!!!' (minimal normalization)
# roberta-base: 'Héllo WORLD!!!'
</syntaxhighlight>

=== Example: Custom Domain-Specific Normalization ===
<syntaxhighlight lang="python">
from tokenizers import normalizers, Tokenizer, pre_tokenizers
from tokenizers.models import BPE

# Create normalizer for code tokenization (preserve case and special chars)
code_normalizer = normalizers.Sequence([
    normalizers.NFD(),  # Only normalize Unicode, preserve case
    normalizers.NFC()   # Recompose to canonical form
])

# For biomedical text (preserve case for acronyms)
bio_normalizer = normalizers.Sequence([
    normalizers.NFD(),
    normalizers.StripAccents()  # Remove accents but keep case
])

# Test code normalizer
code_text = "def calculateSum(arg1, arg2):"
print(f"Code: '{code_normalizer.normalize_str(code_text)}'")
# Output: 'def calculateSum(arg1, arg2):' (case preserved)

# Test bio normalizer
bio_text = "DNA séquence with Protéin"
print(f"Bio: '{bio_normalizer.normalize_str(bio_text)}'")
# Output: 'DNA sequence with Protein' (case preserved, accents removed)
</syntaxhighlight>

=== Example: Replace vs Strip Accents ===
<syntaxhighlight lang="python">
from tokenizers import normalizers

text = "café naïve"

# Method 1: NFD + StripAccents (remove combining marks)
normalizer1 = normalizers.Sequence([
    normalizers.NFD(),
    normalizers.StripAccents()
])
print(f"Strip accents: '{normalizer1.normalize_str(text)}'")
# Output: 'cafe naive'

# Method 2: Replace specific characters (manual approach)
# This would require custom implementation or Replace normalizer
normalizer2 = normalizers.Replace("é", "e")
intermediate = normalizer2.normalize_str(text)
print(f"Replace é→e: '{intermediate}'")
# Output: 'cafe naïve' (only é replaced)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Text_Normalization]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Tokenization_Environment]]
