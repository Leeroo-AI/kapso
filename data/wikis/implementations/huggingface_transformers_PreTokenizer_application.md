{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Tokenizers Pre-tokenizers|https://huggingface.co/docs/tokenizers/api/pre-tokenizers]]
* [[source::Repo|HuggingFace Tokenizers|https://github.com/huggingface/tokenizers]]
|-
! Domains
| [[domain::NLP]], [[domain::Preprocessing]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Concrete tool for applying pre-tokenization splitting using the HuggingFace tokenizers library's PreTokenizer components.

=== Description ===
This implementation wraps the tokenizers library's pre-tokenizer functionality to provide initial text splitting as part of the Transformers tokenization pipeline. The tokenizers library provides efficient Rust-based pre-tokenizer components including Whitespace, ByteLevel, BertPreTokenizer, Metaspace, and others. Pre-tokenizers are automatically applied by fast tokenizers (PreTrainedTokenizerFast) when the configuration is embedded in tokenizer.json. The pre-tokenization step determines which character boundaries cannot be crossed during subsequent subword tokenization.

=== Usage ===
Use this implementation when:
* Working with fast tokenizers that have embedded pre-tokenizer configuration
* Building custom tokenizers with specific splitting requirements
* Debugging tokenization boundary behavior
* Creating domain-specific tokenizers (code, biomedical, social media)
* Understanding how pretrained tokenizers segment text initially

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/tokenizers tokenizers]
* '''File:''' tokenizers library (Rust implementation with Python bindings)
* '''Transformers Integration:''' Fast tokenizers automatically apply pre-tokenizers configured in tokenizer.json

=== Signature ===
<syntaxhighlight lang="python">
# From tokenizers library
from tokenizers import pre_tokenizers

# Common pre-tokenizers
whitespace = pre_tokenizers.Whitespace()
byte_level = pre_tokenizers.ByteLevel()
bert_pre_tokenizer = pre_tokenizers.BertPreTokenizer()
metaspace = pre_tokenizers.Metaspace()

# Compose pre-tokenizers
pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.WhitespaceSplit(),
    pre_tokenizers.Punctuation()
])

# Apply pre-tokenization
pre_tokens = pre_tokenizer.pre_tokenize_str(text)
# Returns: list of (token_str, (start_offset, end_offset))
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# For using pre-tokenizers directly
from tokenizers import pre_tokenizers

# Fast tokenizers apply pre-tokenizers automatically
from transformers import AutoTokenizer
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| text || str || Yes || Normalized input text to pre-tokenize
|-
| pre_tokenizer || pre_tokenizers.PreTokenizer || Yes || Configured pre-tokenizer to apply
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| pre_tokens || list[tuple[str, tuple[int, int]]] || List of (token_string, (start_offset, end_offset)) tuples representing preliminary tokens with character positions
|}

== Usage Examples ==

=== Example: Automatic Pre-tokenization with Fast Tokenizers ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

# Load BERT tokenizer (uses WhitespaceSplit + Punctuation)
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Hello, world! How are you?"

# Pre-tokenization happens automatically during tokenization
tokens = bert_tokenizer.tokenize(text)
print(f"BERT tokens: {tokens}")
# Output: ['hello', ',', 'world', '!', 'how', 'are', 'you', '?']
# Punctuation is split into separate tokens

# Access pre-tokenizer directly (fast tokenizer only)
if hasattr(bert_tokenizer, 'backend_tokenizer'):
    pre_tokenizer = bert_tokenizer.backend_tokenizer.pre_tokenizer
    pre_tokens = pre_tokenizer.pre_tokenize_str(text.lower())
    print(f"Pre-tokens with offsets: {pre_tokens}")
    # Output: [('hello', (0, 5)), (',', (5, 6)), ('world', (7, 12)), ...]
</syntaxhighlight>

=== Example: Comparing Pre-tokenization Strategies ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

text = "Hello, world! 2023"

# BERT: Whitespace + Punctuation splitting
bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
print(f"BERT: {bert_tok.tokenize(text)}")
# Output: ['hello', ',', 'world', '!', '2023']

# GPT-2: Byte-level (preserves spaces as 'Ġ' prefix)
gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
print(f"GPT-2: {gpt2_tok.tokenize(text)}")
# Output: ['Hello', ',', 'Ġworld', '!', 'Ġ2023']
# Note: 'Ġ' represents a leading space in byte-level encoding

# RoBERTa: Byte-level (similar to GPT-2)
roberta_tok = AutoTokenizer.from_pretrained("roberta-base")
print(f"RoBERTa: {roberta_tok.tokenize(text)}")
# Output: ['Hello', ',', 'Ġworld', '!', 'Ġ2023']
</syntaxhighlight>

=== Example: Building Custom Pre-tokenizer ===
<syntaxhighlight lang="python">
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import BPE

# Create tokenizer with custom pre-tokenizer
tokenizer = Tokenizer(BPE())

# Strategy 1: Whitespace only (words + punctuation together)
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

text = "Hello, world!"
pre_tokens = tokenizer.pre_tokenizer.pre_tokenize_str(text)
print(f"Whitespace: {pre_tokens}")
# Output: [('Hello,', (0, 6)), ('world!', (7, 13))]

# Strategy 2: Whitespace + Punctuation
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.WhitespaceSplit(),
    pre_tokenizers.Punctuation()
])

pre_tokens = tokenizer.pre_tokenizer.pre_tokenize_str(text)
print(f"Whitespace + Punct: {pre_tokens}")
# Output: [('Hello', (0, 5)), (',', (5, 6)), ('world', (7, 12)), ('!', (12, 13))]
</syntaxhighlight>

=== Example: Byte-Level Pre-tokenization ===
<syntaxhighlight lang="python">
from tokenizers import pre_tokenizers

# Byte-level pre-tokenizer handles any Unicode
byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False)

# Test with emoji and special characters
text = "Hello 👋 world!"

pre_tokens = byte_level.pre_tokenize_str(text)
print("Byte-level pre-tokens:")
for token, (start, end) in pre_tokens:
    print(f"  '{token}' [{start}:{end}]")

# Output shows byte-level representation:
# 'Hello' [0:5]
# 'Ġ' [5:6]  (space as Ġ)
# '👋' [6:10]  (emoji as byte sequence)
# 'Ġworld' [10:16]
# '!' [16:17]
</syntaxhighlight>

=== Example: Metaspace Pre-tokenization ===
<syntaxhighlight lang="python">
from tokenizers import pre_tokenizers

# Metaspace replaces spaces with '▁' (useful for SentencePiece)
metaspace = pre_tokenizers.Metaspace(replacement='▁', add_prefix_space=True)

text = "Hello world"

pre_tokens = metaspace.pre_tokenize_str(text)
print(f"Metaspace: {pre_tokens}")
# Output: [('▁Hello', (0, 5)), ('▁world', (6, 11))]
# Spaces become ▁ prefix markers
</syntaxhighlight>

=== Example: Digit Splitting Pre-tokenization ===
<syntaxhighlight lang="python">
from tokenizers import pre_tokenizers

# Split digits individually
digits = pre_tokenizers.Digits(individual_digits=True)

text = "In 2023, there were 42 cats"

pre_tokens = digits.pre_tokenize_str(text)
print(f"Digit splitting: {pre_tokens}")
# Output: [('In ', (0, 3)), ('2', (3, 4)), ('0', (4, 5)), ('2', (5, 6)), ('3', (6, 7)),
#          (', there were ', (7, 21)), ('4', (21, 22)), ('2', (22, 23)), (' cats', (23, 28))]
</syntaxhighlight>

=== Example: Custom Domain Pre-tokenizer (Code) ===
<syntaxhighlight lang="python">
from tokenizers import pre_tokenizers, Tokenizer
from tokenizers.models import BPE

# Code-specific pre-tokenizer: preserve identifiers, split operators
code_pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.ByteLevel(add_prefix_space=False),
    pre_tokenizers.Digits(individual_digits=False)  # Keep numbers together
])

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = code_pre_tokenizer

code = "def calculate_sum(x, y): return x + y"

pre_tokens = tokenizer.pre_tokenizer.pre_tokenize_str(code)
print("Code pre-tokens:")
for token, (start, end) in pre_tokens:
    print(f"  '{token}'")
# Output preserves function names and operators as units
</syntaxhighlight>

=== Example: Understanding Boundary Preservation ===
<syntaxhighlight lang="python">
from tokenizers import pre_tokenizers

# Pre-tokenization boundaries are NEVER crossed by subword tokenization
pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.WhitespaceSplit(),
    pre_tokenizers.Punctuation()
])

text = "Hello, world!"

# Pre-tokenization creates boundaries
pre_tokens = pre_tokenizer.pre_tokenize_str(text)
print(f"Pre-token boundaries: {[token for token, _ in pre_tokens]}")
# Output: ['Hello', ',', 'world', '!']

# Subword tokenization will process each pre-token independently
# It can split "Hello" → ["He", "##llo"] but NEVER "Hello," → ["Hel", "##lo,"]
# The comma boundary from pre-tokenization is preserved
</syntaxhighlight>

=== Example: Extracting Character Offsets ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Hello, world! How are you?"

# Get pre-tokens with offsets
if hasattr(tokenizer, 'backend_tokenizer'):
    pre_tokenizer = tokenizer.backend_tokenizer.pre_tokenizer
    pre_tokens = pre_tokenizer.pre_tokenize_str(text)

    print("Pre-tokens with character spans:")
    for token, (start, end) in pre_tokens:
        original_span = text[start:end]
        print(f"  Token: '{token}' | Span: [{start}:{end}] | Original: '{original_span}'")

# Output:
# Token: 'Hello' | Span: [0:5] | Original: 'Hello'
# Token: ',' | Span: [5:6] | Original: ','
# Token: 'world' | Span: [7:12] | Original: 'world'
# ...
# Useful for extractive QA, NER, and span-based tasks
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Pre_Tokenization]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Tokenization_Environment]]
