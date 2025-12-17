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

Concrete tool for adding special tokens to a tokenizer's vocabulary provided by the HuggingFace Transformers library.

=== Description ===

This method adds special tokens (like padding, beginning-of-sequence, end-of-sequence tokens) to the tokenizer and links them to class attributes. If special tokens are not already in the vocabulary, they are added starting from the last vocabulary index. Ensures special tokens receive special handling during tokenization and decoding.

=== Usage ===

Use this when you need to add model-specific special tokens or customize existing ones. Essential when fine-tuning models with new tasks that require additional control tokens.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/tokenization_utils_base.py (lines 1105-1208)

=== Signature ===
<syntaxhighlight lang="python">
def add_special_tokens(
    self,
    special_tokens_dict: dict[str, Union[str, AddedToken, Sequence[Union[str, AddedToken]]]],
    replace_extra_special_tokens: bool = True,
) -> int
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
| special_tokens_dict || dict || Yes || Dictionary mapping special token names to token strings or AddedToken objects
|-
| replace_extra_special_tokens || bool || No || Whether to replace existing extra special tokens or extend them (default: True)
|}

Valid keys for special_tokens_dict:
* bos_token: Beginning of sequence token
* eos_token: End of sequence token
* unk_token: Unknown token for out-of-vocabulary words
* sep_token: Separator token between sequences
* pad_token: Padding token for batch processing
* cls_token: Classification token
* mask_token: Masking token for masked language modeling
* extra_special_tokens: List of additional special tokens

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| num_added || int || Number of tokens added to the vocabulary (0 if all tokens already exist)
|}

== Usage Examples ==

<syntaxhighlight lang="python">
from transformers import AutoTokenizer

# Load base tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Add classification token for GPT-2
special_tokens = {"cls_token": "<CLS>"}
num_added = tokenizer.add_special_tokens(special_tokens)
print(f"Added {num_added} tokens")

# Access new token
print(tokenizer.cls_token)  # "<CLS>"
print(tokenizer.cls_token_id)  # New token ID

# Add multiple special tokens
special_tokens = {
    "pad_token": "[PAD]",
    "sep_token": "[SEP]",
    "extra_special_tokens": ["[TASK1]", "[TASK2]"]
}
num_added = tokenizer.add_special_tokens(special_tokens)

# Important: Resize model embeddings after adding tokens
from transformers import AutoModel
model = AutoModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# Tokens are now accessible
print(tokenizer.pad_token_id)
print(tokenizer.extra_special_tokens)  # ["[TASK1]", "[TASK2]"]
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Special_Tokens]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_PyTorch]]
