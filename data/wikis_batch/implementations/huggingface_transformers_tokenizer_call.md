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

Concrete tool for tokenizing and encoding text sequences into model-ready inputs provided by the HuggingFace Transformers library.

=== Description ===

The main tokenization method that converts text strings into token IDs with optional padding, truncation, and tensor conversion. Returns a BatchEncoding object containing input_ids, attention_mask, and other model inputs. Handles both single sequences and batches efficiently.

=== Usage ===

Use this as the primary method for preparing text data for model input. It combines tokenization, special token addition, padding, truncation, and tensor conversion in one call.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/tokenization_utils_base.py (lines 2471-2596)

=== Signature ===
<syntaxhighlight lang="python">
def __call__(
    self,
    text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput], None] = None,
    text_pair: Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]] = None,
    text_target: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput], None] = None,
    text_pair_target: Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]] = None,
    add_special_tokens: bool = True,
    padding: Union[bool, str, PaddingStrategy] = False,
    truncation: Union[bool, str, TruncationStrategy, None] = None,
    max_length: Optional[int] = None,
    stride: int = 0,
    is_split_into_words: bool = False,
    pad_to_multiple_of: Optional[int] = None,
    padding_side: Optional[str] = None,
    return_tensors: Optional[Union[str, TensorType]] = None,
    return_token_type_ids: Optional[bool] = None,
    return_attention_mask: Optional[bool] = None,
    return_overflowing_tokens: bool = False,
    return_special_tokens_mask: bool = False,
    return_offsets_mapping: bool = False,
    return_length: bool = False,
    verbose: bool = True,
    **kwargs,
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
| text || str or list[str] || Yes* || Single text or batch of texts to encode (*or text_target)
|-
| text_pair || str or list[str] || No || Second sequence for sequence-pair tasks (e.g., QA, NLI)
|-
| add_special_tokens || bool || No || Whether to add model-specific special tokens (default: True)
|-
| padding || bool or str || No || Padding strategy: True/'longest', 'max_length', or False (default: False)
|-
| truncation || bool or str || No || Truncation strategy: True/'longest_first', 'only_first', 'only_second', or False
|-
| max_length || int || No || Maximum sequence length for padding/truncation
|-
| return_tensors || str || No || Return type: 'pt' for PyTorch, 'np' for NumPy (default: None for lists)
|-
| return_attention_mask || bool || No || Whether to return attention mask (default: model-specific)
|-
| is_split_into_words || bool || No || Whether input is pre-tokenized word list (default: False)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| output || BatchEncoding || Dictionary-like object with keys: input_ids, attention_mask, token_type_ids (if applicable)
|}

BatchEncoding fields:
* input_ids: Token IDs for model input
* attention_mask: Mask indicating real tokens (1) vs padding (0)
* token_type_ids: Segment IDs for sequence pairs (BERT-style models)

== Usage Examples ==

<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Basic encoding
text = "Hello, how are you?"
encoded = tokenizer(text)
print(encoded.input_ids)  # [101, 7592, 1010, 2129, 2024, 2017, 1029, 102]

# Batch encoding with padding
texts = ["Short text", "This is a much longer text"]
encoded = tokenizer(texts, padding=True, return_tensors="pt")
print(encoded.input_ids.shape)  # torch.Size([2, max_length])
print(encoded.attention_mask)

# Truncation for long sequences
long_text = "Very " * 1000 + "long text"
encoded = tokenizer(
    long_text,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

# Sequence pairs (e.g., question-answering)
question = "What is the capital of France?"
context = "Paris is the capital and largest city of France."
encoded = tokenizer(
    question,
    context,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

# Pre-tokenized input
words = ["Hello", ",", "world", "!"]
encoded = tokenizer(
    words,
    is_split_into_words=True,
    return_tensors="pt"
)

# Full configuration for model input
encoded = tokenizer(
    text="Sample text",
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt",
    return_attention_mask=True,
    return_token_type_ids=True
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Text_Encoding]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_PyTorch]]
