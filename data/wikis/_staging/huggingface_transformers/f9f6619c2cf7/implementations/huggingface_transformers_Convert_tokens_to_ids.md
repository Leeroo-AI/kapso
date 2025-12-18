{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Tokenizer Documentation|https://huggingface.co/docs/transformers/main_classes/tokenizer]]
|-
! Domains
| [[domain::NLP]], [[domain::Preprocessing]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Concrete tool for converting token strings to vocabulary IDs provided by HuggingFace Transformers.

=== Description ===
This method implements token-to-ID conversion by looking up token strings in the tokenizer's vocabulary dictionary. It handles both single token strings and lists of token strings, automatically falling back to the unknown token ID for tokens not in the vocabulary. The method checks added tokens first (special tokens and user-added tokens) before checking the base vocabulary, ensuring consistent ID assignment for all vocabulary items.

=== Usage ===
Use this implementation when:
* Converting tokenized strings to IDs manually (after custom tokenization)
* Inspecting or debugging vocabulary mappings
* Building custom encoding pipelines
* Verifying token presence in vocabulary
* Creating token ID filters or masks

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/tokenization_utils_base.py:L1478-1492

=== Signature ===
<syntaxhighlight lang="python">
def convert_tokens_to_ids(
    self,
    tokens: Union[str, list[str]]
) -> Union[int, list[int]]
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| tokens || str or list[str] || Yes || Single token string or list of token strings to convert to IDs
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| token_ids || int or list[int] || Single token ID (if input was string) or list of token IDs (if input was list)
|}

== Usage Examples ==

=== Example: Convert Single Token ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Convert single token string to ID
token = "hello"
token_id = tokenizer.convert_tokens_to_ids(token)

print(f"Token: '{token}' -> ID: {token_id}")
# Output: Token: 'hello' -> ID: 7592

# Convert special token
cls_id = tokenizer.convert_tokens_to_ids("[CLS]")
print(f"[CLS] token ID: {cls_id}")
# Output: [CLS] token ID: 101
</syntaxhighlight>

=== Example: Convert List of Tokens ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize text to get tokens
text = "Hello world"
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")
# Output: ['hello', 'world']

# Convert tokens to IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"Token IDs: {token_ids}")
# Output: [7592, 2088]

# This is equivalent to the encode process (without special tokens)
</syntaxhighlight>

=== Example: Unknown Token Handling ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Token that doesn't exist in vocabulary
unknown_token = "xyzabc123notinvocab"
token_id = tokenizer.convert_tokens_to_ids(unknown_token)

print(f"Unknown token: '{unknown_token}' -> ID: {token_id}")
print(f"UNK token ID: {tokenizer.unk_token_id}")
# Output: Both should be 100 (BERT's [UNK] ID)

# Verify it maps to UNK
assert token_id == tokenizer.unk_token_id
</syntaxhighlight>

=== Example: Special Tokens Conversion ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Get all special tokens
special_tokens = tokenizer.all_special_tokens
print(f"Special tokens: {special_tokens}")

# Convert each to ID
print("\nSpecial token IDs:")
for token in special_tokens:
    token_id = tokenizer.convert_tokens_to_ids(token)
    print(f"  {token} -> {token_id}")

# Output:
# [UNK] -> 100
# [CLS] -> 101
# [SEP] -> 102
# [PAD] -> 0
# [MASK] -> 103
</syntaxhighlight>

=== Example: Round-trip Conversion ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Original tokens
tokens = ["hello", "world", "[SEP]"]
print(f"Original tokens: {tokens}")

# Convert to IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"Token IDs: {token_ids}")

# Convert back to tokens
recovered_tokens = tokenizer.convert_ids_to_tokens(token_ids)
print(f"Recovered tokens: {recovered_tokens}")

# Verify round-trip
assert tokens == recovered_tokens
print("Round-trip successful!")
</syntaxhighlight>

=== Example: Subword Token Conversion ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize word that splits into subwords
word = "unbelievable"
tokens = tokenizer.tokenize(word)
print(f"Subword tokens: {tokens}")
# Output: ['un', '##bel', '##iev', '##able']

# Convert each subword to ID
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"Subword IDs: {token_ids}")

# Each subword piece has its own ID
for token, token_id in zip(tokens, token_ids):
    print(f"  '{token}' -> {token_id}")
</syntaxhighlight>

=== Example: Checking Token Existence ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def is_token_in_vocab(token, tokenizer):
    """Check if token exists in vocabulary."""
    token_id = tokenizer.convert_tokens_to_ids(token)
    return token_id != tokenizer.unk_token_id

# Test various tokens
test_tokens = ["hello", "world", "xyznotinvocab", "[CLS]"]

for token in test_tokens:
    in_vocab = is_token_in_vocab(token, tokenizer)
    print(f"'{token}' in vocabulary: {in_vocab}")

# Output:
# 'hello' in vocabulary: True
# 'world' in vocabulary: True
# 'xyznotinvocab' in vocabulary: False
# '[CLS]' in vocabulary: True
</syntaxhighlight>

=== Example: Comparing Different Tokenizers ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

# Load different tokenizers
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

token = "Ġhello"  # GPT-2 uses Ġ prefix for space

# BERT doesn't have this token
bert_id = bert_tokenizer.convert_tokens_to_ids(token)
print(f"BERT: '{token}' -> {bert_id} (UNK: {bert_tokenizer.unk_token_id})")

# GPT-2 has this token
gpt2_id = gpt2_tokenizer.convert_tokens_to_ids(token)
print(f"GPT-2: '{token}' -> {gpt2_id} (UNK: {gpt2_tokenizer.unk_token_id})")

# Same semantic token, different vocabularies
</syntaxhighlight>

=== Example: Manual Encoding Reconstruction ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Hello world"

# Method 1: Use encode (automatic)
auto_ids = tokenizer.encode(text, add_special_tokens=False)
print(f"Automatic encoding: {auto_ids}")

# Method 2: Manual step-by-step
# Step 1: Tokenize
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")

# Step 2: Convert to IDs
manual_ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"Manual encoding: {manual_ids}")

# Should match
assert auto_ids == manual_ids
print("Manual encoding matches automatic!")
</syntaxhighlight>

=== Example: Added Tokens After Loading ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Check initial vocabulary size
print(f"Initial vocab size: {len(tokenizer)}")

# Add custom tokens
custom_tokens = ["<PERSON>", "<LOCATION>", "<ORGANIZATION>"]
num_added = tokenizer.add_tokens(custom_tokens)
print(f"Added {num_added} tokens")

# Convert new tokens to IDs
for token in custom_tokens:
    token_id = tokenizer.convert_tokens_to_ids(token)
    print(f"'{token}' -> ID: {token_id}")

# New tokens have IDs at the end of vocabulary
print(f"New vocab size: {len(tokenizer)}")
</syntaxhighlight>

=== Example: Vocabulary Inspection ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Get full vocabulary
vocab = tokenizer.get_vocab()
print(f"Vocabulary size: {len(vocab)}")

# Find tokens by ID range
print("\nFirst 10 tokens:")
for token_id in range(10):
    token = tokenizer.convert_ids_to_tokens(token_id)
    print(f"  ID {token_id}: '{token}'")

# Find specific tokens
search_tokens = ["##ing", "##ed", "##s"]
print("\nSearching for suffix tokens:")
for token in search_tokens:
    if token in vocab:
        token_id = vocab[token]
        print(f"  '{token}' -> ID {token_id}")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Token_ID_Conversion]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Tokenization_Environment]]
