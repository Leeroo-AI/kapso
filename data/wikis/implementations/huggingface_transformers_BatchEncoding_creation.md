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
Concrete tool for creating BatchEncoding objects that wrap tokenization outputs provided by HuggingFace Transformers.

=== Description ===
This class implements the encoding creation principle by packaging tokenization outputs into a convenient, feature-rich container. BatchEncoding extends Python's dict to provide dictionary-like access, attribute-style access, batch indexing, slicing capabilities, and automatic tensor conversion. It preserves fast tokenizer encoding objects for advanced features like word_ids and character offsets, supports multiple tensor formats (PyTorch, TensorFlow, NumPy), and provides utility methods for device movement and format conversion. The class is returned by tokenizer methods like __call__, encode_plus, and batch_encode_plus.

=== Usage ===
Use this implementation when:
* Returned automatically by tokenizer encoding methods
* Need unified access to tokenization outputs
* Converting between tensor formats
* Using fast tokenizer advanced features (offsets, word_ids)
* Preparing inputs for model forward passes
* Implementing custom tokenization workflows

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/tokenization_utils_base.py:L200-350

=== Signature ===
<syntaxhighlight lang="python">
class BatchEncoding(dict):
    def __init__(
        self,
        data: Optional[dict[str, Any]] = None,
        encoding: Optional[Union[EncodingFast, Sequence[EncodingFast]]] = None,
        tensor_type: Union[None, str, TensorType] = None,
        prepend_batch_axis: bool = False,
        n_sequences: Optional[int] = None,
    )
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer
# BatchEncoding is returned by tokenizer methods
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| data || dict[str, Any] || No || Dictionary containing input_ids, attention_mask, token_type_ids, etc.
|-
| encoding || EncodingFast or Sequence[EncodingFast] || No || Fast tokenizer encoding objects with advanced features
|-
| tensor_type || str or TensorType || No || Tensor format: 'pt' (PyTorch), 'tf' (TensorFlow), 'np' (NumPy), or None (lists)
|-
| prepend_batch_axis || bool || No || Whether to add batch dimension when converting to tensors (default: False)
|-
| n_sequences || int || No || Number of sequences: 1 (single), 2 (pair), or None (unknown)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| batch_encoding || BatchEncoding || Dict-like object with tokenization outputs and convenience methods
|}

== Usage Examples ==

=== Example: Basic BatchEncoding Access ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Encode text - returns BatchEncoding
encoding = tokenizer("Hello, how are you?")

print(f"Type: {type(encoding)}")
# Output: <class 'transformers.tokenization_utils_base.BatchEncoding'>

# Access methods:
# 1. Dict-style
print(f"Dict access: {encoding['input_ids']}")

# 2. Attribute-style
print(f"Attribute access: {encoding.input_ids}")

# 3. Keys iteration
print(f"Keys: {list(encoding.keys())}")
# Output: ['input_ids', 'token_type_ids', 'attention_mask']

# 4. Items iteration
for key, value in encoding.items():
    print(f"{key}: {value}")
</syntaxhighlight>

=== Example: Tensor Conversion ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Hello world"

# Return as Python lists (default)
list_encoding = tokenizer(text)
print(f"List type: {type(list_encoding['input_ids'])}")
# Output: <class 'list'>

# Return as PyTorch tensors
pt_encoding = tokenizer(text, return_tensors='pt')
print(f"PyTorch type: {type(pt_encoding['input_ids'])}")
print(f"PyTorch shape: {pt_encoding['input_ids'].shape}")
# Output: torch.Size([1, 6])

# Return as NumPy arrays
np_encoding = tokenizer(text, return_tensors='np')
print(f"NumPy type: {type(np_encoding['input_ids'])}")
print(f"NumPy shape: {np_encoding['input_ids'].shape}")

# Return as TensorFlow tensors
try:
    tf_encoding = tokenizer(text, return_tensors='tf')
    print(f"TensorFlow type: {type(tf_encoding['input_ids'])}")
except:
    print("TensorFlow not available")
</syntaxhighlight>

=== Example: Batch Indexing and Slicing ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

texts = ["First sentence", "Second sentence", "Third sentence"]

# Batch encode
batch_encoding = tokenizer(texts, padding=True, return_tensors='pt')

print(f"Batch shape: {batch_encoding['input_ids'].shape}")
# Output: torch.Size([3, 5])

# Index single sequence
first_seq = batch_encoding[0]
print(f"First sequence encoding type: {type(first_seq)}")
# Returns encoding for first sequence

# Slice batch
first_two = batch_encoding[:2]
print(f"Sliced keys: {first_two.keys()}")
print(f"Sliced shape: {first_two['input_ids'].shape}")
# Output: torch.Size([2, 5])

# Dictionary-style slicing also works
input_ids_slice = batch_encoding['input_ids'][:2]
print(f"Direct slice shape: {input_ids_slice.shape}")
</syntaxhighlight>

=== Example: Fast Tokenizer Advanced Features ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

# Load fast tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Hello world"
encoding = tokenizer(text)

# Check if fast tokenizer
print(f"Is fast: {encoding.is_fast}")

if encoding.is_fast:
    # Get token strings
    tokens = encoding.tokens()
    print(f"Tokens: {tokens}")
    # Output: ['[CLS]', 'hello', 'world', '[SEP]']

    # Get word IDs (which word each token belongs to)
    word_ids = encoding.word_ids()
    print(f"Word IDs: {word_ids}")
    # Output: [None, 0, 1, None]
    # None for special tokens, 0 for first word, 1 for second word

    # Get sequence IDs (for sequence pairs)
    sequence_ids = encoding.sequence_ids()
    print(f"Sequence IDs: {sequence_ids}")
    # Output: [None, 0, 0, None]
    # 0 for first sequence, 1 for second (if pair), None for special
</syntaxhighlight>

=== Example: Character Offsets (Extractive QA) ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Hello world! How are you?"
encoding = tokenizer(text, return_offsets_mapping=True)

print("Token to character mapping:")
for i, (token, (start, end)) in enumerate(zip(
    encoding.tokens(),
    encoding['offset_mapping']
)):
    if start is not None and end is not None:
        original_span = text[start:end]
        print(f"Token {i}: '{token}' -> chars [{start}:{end}] '{original_span}'")

# Output:
# Token 0: '[CLS]' -> chars [0:0] ''
# Token 1: 'hello' -> chars [0:5] 'Hello'
# Token 2: 'world' -> chars [6:11] 'world'
# ...
# Useful for span-based tasks like question answering
</syntaxhighlight>

=== Example: Device Movement (PyTorch) ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Hello world"
encoding = tokenizer(text, return_tensors='pt')

print(f"Initial device: {encoding['input_ids'].device}")
# Output: cpu

# Move to GPU if available
if torch.cuda.is_available():
    encoding = encoding.to('cuda')
    print(f"After to('cuda'): {encoding['input_ids'].device}")

# Move back to CPU
encoding = encoding.to('cpu')
print(f"After to('cpu'): {encoding['input_ids'].device}")

# Can also use device objects
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoding = encoding.to(device)
</syntaxhighlight>

=== Example: Model Input Preparation ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

text = "Hello, how are you?"

# Encode with proper format
encoding = tokenizer(text, return_tensors='pt')

print(f"Encoding keys: {encoding.keys()}")
# Output: dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])

# Pass directly to model using ** unpacking
outputs = model(**encoding)

print(f"Model output shape: {outputs.last_hidden_state.shape}")
# Output: torch.Size([1, 7, 768])
# [batch_size, sequence_length, hidden_size]

# BatchEncoding makes model input preparation seamless
</syntaxhighlight>

=== Example: Sequence Pairs ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Encode sequence pair
question = "What is the capital?"
context = "Paris is the capital of France."

encoding = tokenizer(question, context, return_tensors='pt')

print(f"Number of sequences: {encoding.n_sequences}")
# Output: 2

print(f"Input IDs shape: {encoding['input_ids'].shape}")
print(f"Token type IDs: {encoding['token_type_ids']}")
# Token type IDs: 0 for question, 1 for context

# Get sequence IDs for each token
if encoding.is_fast:
    seq_ids = encoding.sequence_ids()
    print(f"Sequence IDs: {seq_ids}")
    # [None, 0, 0, 0, 0, None, 1, 1, 1, 1, 1, 1, None]
</syntaxhighlight>

=== Example: Converting Tensor Types ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Hello world"

# Start with PyTorch
pt_encoding = tokenizer(text, return_tensors='pt')
print(f"PyTorch: {type(pt_encoding['input_ids'])}")

# Convert to list
list_encoding = pt_encoding.to_dict()
print(f"To dict: {type(list_encoding)}")

# Convert to NumPy
np_array = pt_encoding['input_ids'].numpy()
print(f"NumPy: {type(np_array)}")

# Convert back to list
py_list = pt_encoding['input_ids'].tolist()
print(f"Python list: {type(py_list)}")
</syntaxhighlight>

=== Example: Batch Processing with BatchEncoding ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

texts = [
    "First example",
    "Second example",
    "Third example"
]

# Encode batch
batch_encoding = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

print(f"Batch size: {batch_encoding['input_ids'].shape[0]}")
print(f"Sequence length: {batch_encoding['input_ids'].shape[1]}")

# Process batch through model
with torch.no_grad():
    outputs = model(**batch_encoding)

print(f"Output shape: {outputs.last_hidden_state.shape}")
# torch.Size([3, max_len, 768])

# Extract embeddings for each sequence
for i in range(len(texts)):
    # Use CLS token embedding as sequence representation
    cls_embedding = outputs.last_hidden_state[i, 0, :]
    print(f"Sequence {i} embedding shape: {cls_embedding.shape}")
</syntaxhighlight>

=== Example: Overflowing Tokens (Stride) ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Long text that will be split
long_text = "This is a very long text. " * 50

# Encode with stride (for question answering on long documents)
encoding = tokenizer(
    long_text,
    truncation=True,
    max_length=128,
    stride=20,
    return_overflowing_tokens=True,
    return_tensors='pt'
)

print(f"Number of chunks: {encoding['input_ids'].shape[0]}")
print(f"Chunk size: {encoding['input_ids'].shape[1]}")

# Each chunk overlaps by stride tokens
# Useful for processing long documents without losing context
</syntaxhighlight>

=== Example: Custom BatchEncoding Creation ===
<syntaxhighlight lang="python">
from transformers.tokenization_utils_base import BatchEncoding

# Create custom BatchEncoding
data = {
    'input_ids': [[101, 2023, 2003, 102]],
    'attention_mask': [[1, 1, 1, 1]],
    'token_type_ids': [[0, 0, 0, 0]]
}

# Create without tensor conversion
custom_encoding = BatchEncoding(data)

print(f"Type: {type(custom_encoding)}")
print(f"Keys: {list(custom_encoding.keys())}")
print(f"Input IDs: {custom_encoding.input_ids}")

# Convert to tensors
import torch
custom_encoding = BatchEncoding(data, tensor_type='pt')
print(f"Tensor type: {type(custom_encoding.input_ids)}")
print(f"Shape: {custom_encoding.input_ids.shape}")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Encoding_Creation]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Tokenization_Environment]]
