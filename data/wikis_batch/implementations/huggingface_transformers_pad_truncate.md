{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Tokenization]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Concrete tool for padding encoded token sequences to uniform length provided by the HuggingFace Transformers library.

=== Description ===

This method pads already-encoded inputs to a specified or maximum length. It can pad single encoded inputs or batches, adding padding tokens on the left or right side. Handles attention masks, token type IDs, and special tokens masks appropriately during padding. Supports padding to multiples of specific values for hardware optimization.

=== Usage ===

Use this when you need to pad pre-encoded sequences, especially in DataLoader collate functions or when working with batches of varying lengths. Useful for dynamic batching scenarios.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/tokenization_utils_base.py (lines 2623-2794)
* '''Internal method:''' _pad (lines 2796-2876)

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
| encoded_inputs || BatchEncoding, list, or dict || Yes || Already tokenized inputs to pad
|-
| padding || bool or str || No || Padding strategy: True/'longest', 'max_length', False (default: True)
|-
| max_length || int || No || Maximum length to pad to (uses model max if not specified)
|-
| pad_to_multiple_of || int || No || Pad length to be multiple of this value (for Tensor Cores)
|-
| padding_side || str || No || Side to pad: 'right' or 'left' (default: tokenizer's default)
|-
| return_attention_mask || bool || No || Whether to return attention mask
|-
| return_tensors || str || No || Convert to tensors: 'pt' (PyTorch) or 'np' (NumPy)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| padded_inputs || BatchEncoding || Padded sequences with attention masks and other relevant fields
|}

== Usage Examples ==

<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Encode without padding first
text1 = tokenizer("Short", add_special_tokens=True)
text2 = tokenizer("This is longer text", add_special_tokens=True)

# Pad manually later
batch = [text1, text2]
padded = tokenizer.pad(batch, padding=True, return_tensors="pt")
print(padded.input_ids.shape)  # All sequences same length
print(padded.attention_mask)   # Padding positions marked as 0

# Pad to specific max length
padded = tokenizer.pad(
    batch,
    padding="max_length",
    max_length=20,
    return_tensors="pt"
)

# Pad to multiple of 8 (for GPU optimization)
padded = tokenizer.pad(
    batch,
    padding=True,
    pad_to_multiple_of=8,
    return_tensors="pt"
)

# Left padding (useful for generation tasks)
tokenizer.padding_side = "left"
padded = tokenizer.pad(batch, padding=True, return_tensors="pt")

# Use in PyTorch DataLoader collate function
from torch.utils.data import DataLoader

def collate_fn(batch):
    # batch is list of encoded examples
    return tokenizer.pad(
        batch,
        padding=True,
        return_tensors="pt"
    )

# Example with dict format
encoded_dict = {
    "input_ids": [[101, 2023, 102], [101, 7592, 1010, 102]],
    "attention_mask": [[1, 1, 1], [1, 1, 1, 1]]
}
padded = tokenizer.pad(encoded_dict, padding=True, return_tensors="pt")

# Handling list of dicts (common in datasets)
batch_of_dicts = [
    {"input_ids": [101, 2023, 102]},
    {"input_ids": [101, 7592, 1010, 2129, 102]},
]
padded = tokenizer.pad(batch_of_dicts, padding=True, return_tensors="pt")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Padding_Truncation]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_PyTorch]]
