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

Concrete tool for converting token IDs back into human-readable text strings provided by the HuggingFace Transformers library.

=== Description ===

This method converts sequences of token IDs back into text strings. It handles both single sequences and batches, optionally removes special tokens (like padding, BOS, EOS), and cleans up tokenization artifacts (extra spaces, subword indicators). Essential for interpreting model outputs and generated text.

=== Usage ===

Use this to convert model-generated token IDs back into readable text. Critical for text generation tasks, translation, summarization, and any scenario where you need to interpret tokenized outputs.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/tokenization_utils_base.py (lines 2891-2937)
* '''Batch variant:''' batch_decode (lines 2939-2976)
* '''Internal method:''' _decode (lines 2978-2985)

=== Signature ===
<syntaxhighlight lang="python">
def decode(
    self,
    token_ids: Union[int, list[int], list[list[int]], np.ndarray, torch.Tensor],
    skip_special_tokens: bool = False,
    **kwargs,
) -> Union[str, list[str]]
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
| token_ids || int, list[int], list[list[int]], np.ndarray, or torch.Tensor || Yes || Token IDs to decode (single sequence or batch)
|-
| skip_special_tokens || bool || No || Whether to remove special tokens from output (default: False)
|-
| clean_up_tokenization_spaces || bool || No || Whether to clean extra spaces (default: model-specific)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| text || str or list[str] || Decoded text string(s)
|}

== Usage Examples ==

<syntaxhighlight lang="python">
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Decode single sequence
token_ids = [101, 7592, 1010, 2129, 2024, 2017, 1029, 102]
text = tokenizer.decode(token_ids)
print(text)  # "[CLS] hello, how are you? [SEP]"

# Skip special tokens
text = tokenizer.decode(token_ids, skip_special_tokens=True)
print(text)  # "hello, how are you?"

# Decode batch of sequences
batch_ids = [
    [101, 7592, 102],
    [101, 2129, 2024, 2017, 102]
]
texts = tokenizer.decode(batch_ids)
print(texts)  # ["[CLS] hello [SEP]", "[CLS] how are you [SEP]"]

# Using batch_decode (equivalent)
texts = tokenizer.batch_decode(batch_ids, skip_special_tokens=True)
print(texts)  # ["hello", "how are you"]

# Decode model generation output
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate
outputs = model.generate(
    inputs.input_ids,
    max_new_tokens=50,
    num_return_sequences=3
)

# Decode generated sequences
for i, output in enumerate(outputs):
    text = tokenizer.decode(output, skip_special_tokens=True)
    print(f"Generation {i}: {text}")

# Decode with PyTorch tensors
token_tensor = torch.tensor([101, 7592, 102])
text = tokenizer.decode(token_tensor)

# Decode with NumPy arrays
import numpy as np
token_array = np.array([101, 7592, 102])
text = tokenizer.decode(token_array)

# Handle padding in batch decoding
tokenizer.pad_token = tokenizer.eos_token
batch_ids = [
    [101, 7592, 102, 0, 0],  # padded
    [101, 2129, 2024, 2017, 102]
]
texts = tokenizer.batch_decode(batch_ids, skip_special_tokens=True)

# Clean up tokenization spaces (for models with subword tokenization)
tokenizer_bpe = AutoTokenizer.from_pretrained("gpt2")
token_ids = [15496, 995]  # "Hello" + "world"
text = tokenizer_bpe.decode(token_ids, clean_up_tokenization_spaces=True)
print(text)  # "Helloworld" -> cleaned to "Hello world"

# Streaming decoding (decode incrementally)
generated_ids = []
for new_token_id in [15496, 995, 0]:  # simulate generation
    generated_ids.append(new_token_id)
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Current: {text}")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Text_Decoding]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_PyTorch]]
