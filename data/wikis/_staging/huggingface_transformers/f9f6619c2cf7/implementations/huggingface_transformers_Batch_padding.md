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
Concrete tool for padding and truncating token sequences to uniform lengths provided by HuggingFace Transformers.

=== Description ===
This method implements padding and truncation by taking encoded inputs (single sequences or batches) and ensuring they meet length requirements. It handles multiple input formats (BatchEncoding, dicts, lists), applies padding to the specified side (left or right), generates attention masks to mark real vs padded tokens, optionally converts to tensors (PyTorch, TensorFlow, NumPy), and supports padding to multiples for hardware efficiency. The pad method is essential for batch processing and can be used standalone or automatically via tokenizer __call__.

=== Usage ===
Use this implementation when:
* Creating batches of variable-length sequences
* Preparing inputs for batch inference
* Implementing custom data collators for training
* Post-processing encoded sequences before model input
* Converting between different tensor formats

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/tokenization_utils_base.py:L2623-2700

=== Signature ===
<syntaxhighlight lang="python">
def pad(
    self,
    encoded_inputs: Union[
        BatchEncoding,
        list[BatchEncoding],
        dict[str, EncodedInput],
        dict[str, list[EncodedInput]],
        list[dict[str, EncodedInput]],
    ],
    padding: Union[bool, str, PaddingStrategy] = True,
    max_length: Optional[int] = None,
    pad_to_multiple_of: Optional[int] = None,
    padding_side: Optional[str] = None,
    return_attention_mask: Optional[bool] = None,
    return_tensors: Optional[Union[str, TensorType]] = None,
    verbose: bool = True,
) -> BatchEncoding
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
| encoded_inputs || BatchEncoding, list, or dict || Yes || Tokenized inputs to pad (single sequence or batch)
|-
| padding || bool, str, or PaddingStrategy || No || Padding strategy: True/'longest', 'max_length', False/'do_not_pad' (default: True)
|-
| max_length || int || No || Maximum length for padding (required if padding='max_length')
|-
| pad_to_multiple_of || int || No || Pad to multiple of this value for hardware efficiency (e.g., 8 for Tensor Cores)
|-
| padding_side || str || No || Which side to pad: 'right' or 'left' (defaults to tokenizer's padding_side)
|-
| return_attention_mask || bool || No || Whether to return attention mask (defaults to model specifics)
|-
| return_tensors || str or TensorType || No || Return as 'pt' (PyTorch), 'tf' (TensorFlow), or 'np' (NumPy) (default: None)
|-
| verbose || bool || No || Whether to print warnings (default: True)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| padded_encoding || BatchEncoding || Padded sequences with input_ids, attention_mask, and optionally token_type_ids
|}

== Usage Examples ==

=== Example: Basic Padding ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Encode multiple sequences (different lengths)
texts = ["Hello", "Hello world", "Hello world how are you"]

# Encode without padding
encoded = [tokenizer.encode(text, add_special_tokens=True) for text in texts]
print("Unpadded lengths:", [len(ids) for ids in encoded])
# Output: [3, 4, 7]

# Pad to longest in batch
padded = tokenizer.pad(
    {"input_ids": encoded},
    padding=True,
    return_attention_mask=True
)

print("Padded input_ids shape:", len(padded['input_ids']), "x", len(padded['input_ids'][0]))
print("Padded sequences:")
for ids, mask in zip(padded['input_ids'], padded['attention_mask']):
    print(f"  IDs: {ids}")
    print(f"  Mask: {mask}")

# Output:
# All sequences now length 7 (longest)
# Padding tokens (0) added at end
# Attention masks mark real (1) vs padding (0) tokens
</syntaxhighlight>

=== Example: Padding Strategies ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

texts = ["Short", "This is a longer sentence"]

# Strategy 1: Pad to longest (dynamic)
encoded1 = tokenizer(texts, padding=True)
print(f"Longest: shape {len(encoded1['input_ids'][0])}")
# Pads to length of longest sequence in batch

# Strategy 2: Pad to max_length (fixed)
encoded2 = tokenizer(texts, padding='max_length', max_length=20)
print(f"Max length: shape {len(encoded2['input_ids'][0])}")
# All sequences padded to exactly 20 tokens

# Strategy 3: No padding
encoded3 = tokenizer(texts, padding=False)
print(f"No padding: lengths {[len(ids) for ids in encoded3['input_ids']]}")
# Sequences remain different lengths
</syntaxhighlight>

=== Example: Left vs Right Padding ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

# BERT uses right padding by default
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# GPT-2 should use left padding for generation
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token  # GPT-2 needs pad token set

text = "Hello world"

# Right padding (BERT style)
right_padded = bert_tokenizer(text, padding='max_length', max_length=10)
print(f"Right padding: {right_padded['input_ids']}")
# Output: [101, 7592, 2088, 102, 0, 0, 0, 0, 0, 0]
# Padding at end

# Left padding (GPT style for generation)
gpt2_tokenizer.padding_side = "left"
left_padded = gpt2_tokenizer(text, padding='max_length', max_length=10)
print(f"Left padding: {left_padded['input_ids']}")
# Output: [50256, 50256, 50256, ..., 15496, 995]
# Padding at start
</syntaxhighlight>

=== Example: Padding with Truncation ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

texts = [
    "Short",
    "Medium length sentence here",
    "This is a very long sentence that will definitely exceed our maximum length limit"
]

# Encode with both padding and truncation
encoded = tokenizer(
    texts,
    padding='max_length',
    truncation=True,
    max_length=15,
    return_tensors='pt'
)

print(f"Shape: {encoded['input_ids'].shape}")
# Output: torch.Size([3, 15])
# All sequences exactly 15 tokens

print("\nSequences:")
for i, (ids, mask) in enumerate(zip(encoded['input_ids'], encoded['attention_mask'])):
    num_real = mask.sum().item()
    print(f"Text {i}: {num_real} real tokens, {15 - num_real} padding")
</syntaxhighlight>

=== Example: Pad to Multiple (Hardware Optimization) ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

texts = ["Hello", "Hello world how are you doing today"]

# Pad to multiple of 8 for Tensor Core efficiency
encoded = tokenizer(
    texts,
    padding=True,
    pad_to_multiple_of=8,
    return_attention_mask=True
)

print("Sequence lengths:")
for i, ids in enumerate(encoded['input_ids']):
    print(f"  Sequence {i}: length {len(ids)} (multiple of 8)")

# Output:
# Sequence 0: length 8
# Sequence 1: length 16
# Padded to next multiple of 8 for GPU efficiency
</syntaxhighlight>

=== Example: Custom Data Collator ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Return unpadded encoding
        return tokenizer(self.texts[idx], truncation=True, max_length=128)

# Custom collator using pad()
def collate_fn(batch):
    # batch is list of encodings with different lengths
    # Pad them to same length
    return tokenizer.pad(
        batch,
        padding=True,
        return_tensors='pt'
    )

# Create dataset and dataloader
texts = ["Short", "Medium length", "Very long sentence here"]
dataset = TextDataset(texts)
dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

# Iterate - each batch is padded
for batch in dataloader:
    print(f"Batch shape: {batch['input_ids'].shape}")
    print(f"Attention mask: {batch['attention_mask'].shape}")
</syntaxhighlight>

=== Example: Padding Pre-encoded Sequences ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Pre-encode sequences separately
ids1 = tokenizer.encode("Hello", add_special_tokens=True)
ids2 = tokenizer.encode("Hello world", add_special_tokens=True)
ids3 = tokenizer.encode("Hello world today", add_special_tokens=True)

print("Original lengths:", [len(ids1), len(ids2), len(ids3)])

# Pad them using pad() method
padded = tokenizer.pad(
    {"input_ids": [ids1, ids2, ids3]},
    padding=True,
    return_tensors='pt'
)

print(f"Padded shape: {padded['input_ids'].shape}")
print(f"Attention mask shape: {padded['attention_mask'].shape}")

# Can now use as model input
# output = model(**padded)
</syntaxhighlight>

=== Example: Different Tensor Types ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

texts = ["Hello", "Hello world"]

# Return PyTorch tensors
pt_encoded = tokenizer(texts, padding=True, return_tensors='pt')
print(f"PyTorch type: {type(pt_encoded['input_ids'])}")
print(f"PyTorch shape: {pt_encoded['input_ids'].shape}")

# Return NumPy arrays
np_encoded = tokenizer(texts, padding=True, return_tensors='np')
print(f"\nNumPy type: {type(np_encoded['input_ids'])}")
print(f"NumPy shape: {np_encoded['input_ids'].shape}")

# Return TensorFlow tensors (if TensorFlow installed)
try:
    tf_encoded = tokenizer(texts, padding=True, return_tensors='tf')
    print(f"\nTensorFlow type: {type(tf_encoded['input_ids'])}")
    print(f"TensorFlow shape: {tf_encoded['input_ids'].shape}")
except Exception as e:
    print(f"\nTensorFlow not available: {e}")
</syntaxhighlight>

=== Example: Padding with Token Type IDs ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Encode sequence pairs (creates token_type_ids)
pairs = [
    ("Question one", "Context one"),
    ("Question two is longer", "Context two")
]

encoded = tokenizer(
    [q for q, _ in pairs],
    [c for _, c in pairs],
    padding=True,
    return_tensors='pt'
)

print(f"Input IDs shape: {encoded['input_ids'].shape}")
print(f"Token type IDs shape: {encoded['token_type_ids'].shape}")
print(f"Attention mask shape: {encoded['attention_mask'].shape}")

# Token type IDs are also padded
print("\nToken type IDs for first pair:")
print(encoded['token_type_ids'][0])
# 0s for question, 1s for context, 0s for padding
</syntaxhighlight>

=== Example: Handling Edge Cases ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Edge case 1: Empty sequence
empty_encoded = tokenizer("", padding='max_length', max_length=10)
print(f"Empty sequence: {empty_encoded['input_ids']}")
# Just special tokens + padding

# Edge case 2: All sequences already same length
same_length = ["Hello", "World"]
encoded = tokenizer(same_length, add_special_tokens=False)
padded = tokenizer.pad(encoded, padding=True)
print(f"Already same length: {[len(ids) for ids in padded['input_ids']]}")
# No padding needed

# Edge case 3: Single sequence (no batching)
single = tokenizer("Hello", padding='max_length', max_length=10)
print(f"Single sequence: {len(single['input_ids'])}")
# Still padded to max_length
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Padding_Truncation]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Tokenization_Environment]]
