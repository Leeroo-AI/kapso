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
Concrete tool for encoding text into token ID sequences provided by HuggingFace Transformers.

=== Description ===
This method implements the subword tokenization principle by converting input text (or text pairs) into lists of integer token IDs. It orchestrates the complete tokenization pipeline: normalizing text, pre-tokenizing, applying subword segmentation, looking up vocabulary IDs, adding special tokens, and optionally applying padding/truncation. The encode method is the primary entry point for converting raw text into model-ready input IDs, returning just the token IDs without additional metadata like attention masks.

=== Usage ===
Use this implementation when:
* Converting text to token IDs for model inference
* Encoding single sequences or sequence pairs
* Need only token IDs without attention masks or other metadata
* Preprocessing text for embedding lookup
* Batch processing where padding will be applied separately

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/tokenization_utils_base.py:L2294-2345

=== Signature ===
<syntaxhighlight lang="python">
def encode(
    self,
    text: Union[TextInput, PreTokenizedInput, EncodedInput],
    text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
    add_special_tokens: bool = True,
    padding: Union[bool, str, PaddingStrategy] = False,
    truncation: Union[bool, str, TruncationStrategy, None] = None,
    max_length: Optional[int] = None,
    stride: int = 0,
    padding_side: Optional[str] = None,
    return_tensors: Optional[Union[str, TensorType]] = None,
    **kwargs,
) -> list[int]
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
| text || str, list[str], or list[int] || Yes || Input text to encode (or pre-tokenized tokens, or pre-encoded IDs)
|-
| text_pair || str, list[str], or list[int] || No || Optional second sequence for sequence-pair tasks (e.g., QA, NLI)
|-
| add_special_tokens || bool || No || Whether to add model-specific special tokens (CLS, SEP, etc.) (default: True)
|-
| padding || bool, str, or PaddingStrategy || No || Padding strategy: True/'longest', 'max_length', or False/'do_not_pad' (default: False)
|-
| truncation || bool, str, or TruncationStrategy || No || Truncation strategy: True, 'longest_first', 'only_first', 'only_second', or None (default: None)
|-
| max_length || int || No || Maximum sequence length for padding/truncation (defaults to model max length)
|-
| stride || int || No || Stride for truncation when returning overflowing tokens (default: 0)
|-
| padding_side || str || No || Which side to pad: 'right' or 'left' (defaults to tokenizer's padding_side)
|-
| return_tensors || str or TensorType || No || Return as 'pt' (PyTorch), 'tf' (TensorFlow), or 'np' (NumPy) (default: None, returns list)
|-
| **kwargs || dict || No || Additional arguments passed to internal methods
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| input_ids || list[int] || List of token IDs representing the encoded text
|}

== Usage Examples ==

=== Example: Basic Text Encoding ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Encode single sequence
text = "Hello, how are you?"
input_ids = tokenizer.encode(text)

print(f"Text: {text}")
print(f"Token IDs: {input_ids}")
# Output: [101, 7592, 1010, 2129, 2024, 2017, 1029, 102]

# Decode back to text
decoded = tokenizer.decode(input_ids)
print(f"Decoded: {decoded}")
# Output: [CLS] hello, how are you? [SEP]
</syntaxhighlight>

=== Example: Encoding Sequence Pairs ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Encode sequence pair (e.g., for question answering)
question = "What is the capital of France?"
context = "Paris is the capital of France."

input_ids = tokenizer.encode(question, context)

print(f"Question: {question}")
print(f"Context: {context}")
print(f"Encoded IDs: {input_ids}")

# Decode to see structure
decoded = tokenizer.decode(input_ids)
print(f"Decoded: {decoded}")
# Output: [CLS] what is the capital of france? [SEP] paris is the capital of france. [SEP]
</syntaxhighlight>

=== Example: Without Special Tokens ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "Hello world"

# With special tokens (default)
with_special = tokenizer.encode(text, add_special_tokens=True)
print(f"With special tokens: {with_special}")

# Without special tokens
without_special = tokenizer.encode(text, add_special_tokens=False)
print(f"Without special tokens: {without_special}")

# Decode both
print(f"With: '{tokenizer.decode(with_special)}'")
print(f"Without: '{tokenizer.decode(without_special)}'")
</syntaxhighlight>

=== Example: Truncation and Max Length ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

long_text = "This is a very long sentence " * 50  # Create very long text

# Without truncation (may exceed model max length)
try:
    full_ids = tokenizer.encode(long_text, truncation=False)
    print(f"Full length: {len(full_ids)}")
except:
    print("Text too long without truncation!")

# With truncation to 20 tokens
truncated_ids = tokenizer.encode(long_text, truncation=True, max_length=20)
print(f"Truncated length: {len(truncated_ids)}")
print(f"Truncated IDs: {truncated_ids}")

# Special tokens are preserved after truncation
decoded = tokenizer.decode(truncated_ids)
print(f"Decoded: {decoded}")
# Output includes [CLS] at start and [SEP] at end
</syntaxhighlight>

=== Example: Different Tokenization Strategies ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

text = "unbelievable"

# WordPiece (BERT) - uses ## for subwords
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_ids = bert_tokenizer.encode(text, add_special_tokens=False)
bert_tokens = bert_tokenizer.tokenize(text)
print(f"BERT tokens: {bert_tokens}")  # ['un', '##bel', '##iev', '##able']
print(f"BERT IDs: {bert_ids}")

# BPE (GPT-2) - no special prefix
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt2_ids = gpt2_tokenizer.encode(text, add_special_tokens=False)
gpt2_tokens = gpt2_tokenizer.tokenize(text)
print(f"GPT-2 tokens: {gpt2_tokens}")  # ['un', 'bel', 'iev', 'able']
print(f"GPT-2 IDs: {gpt2_ids}")
</syntaxhighlight>

=== Example: Handling Unknown Tokens ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Text with very rare/unknown word
text = "Hello xyzabc123"  # xyzabc123 likely not in vocabulary

input_ids = tokenizer.encode(text)
tokens = tokenizer.tokenize(text)

print(f"Tokens: {tokens}")
# Output: ['hello', 'xyz', '##ab', '##c', '##12', '##3'] or similar subword split

# The unknown word is split into subwords
# If a piece is truly unknown, it becomes [UNK]
very_weird_text = "Hello \u0001\u0002\u0003"  # Control characters
tokens = tokenizer.tokenize(very_weird_text)
print(f"Weird tokens: {tokens}")
</syntaxhighlight>

=== Example: Batch Encoding vs Single Encoding ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

texts = [
    "First sentence",
    "Second sentence",
    "Third sentence"
]

# Method 1: Encode each separately
individual_ids = [tokenizer.encode(text) for text in texts]
print("Individual encoding:")
for i, ids in enumerate(individual_ids):
    print(f"  Text {i}: length {len(ids)}")

# Method 2: Use __call__ for batch (better for padding)
batch_encoding = tokenizer(texts, padding=True, return_tensors="pt")
print(f"\nBatch encoding shape: {batch_encoding['input_ids'].shape}")

# encode() returns just IDs, __call__ returns dict with attention_mask, etc.
</syntaxhighlight>

=== Example: Pre-tokenized Input ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Can pass pre-tokenized words
tokens = ["Hello", "world"]
input_ids = tokenizer.encode(tokens, is_split_into_words=True)

print(f"Tokens: {tokens}")
print(f"Encoded: {input_ids}")

# Tokenizer will still apply subword tokenization to each word
decoded = tokenizer.decode(input_ids)
print(f"Decoded: {decoded}")
</syntaxhighlight>

=== Example: Encode vs __call__ Comparison ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Hello world"

# encode() returns only input_ids as list
input_ids = tokenizer.encode(text)
print(f"encode() output type: {type(input_ids)}")
print(f"encode() output: {input_ids}")

# __call__() returns BatchEncoding with metadata
encoding = tokenizer(text)
print(f"\n__call__() output type: {type(encoding)}")
print(f"__call__() keys: {encoding.keys()}")
print(f"  input_ids: {encoding['input_ids']}")
print(f"  attention_mask: {encoding['attention_mask']}")

# Use encode() when you only need IDs
# Use __call__() when you need full BatchEncoding with masks
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Subword_Tokenization]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Tokenization_Environment]]
