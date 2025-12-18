{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Trainer Documentation|https://huggingface.co/docs/transformers/main_classes/trainer]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Concrete implementation for dynamic batch creation and padding provided by HuggingFace Transformers.

=== Description ===
The DataCollatorWithPadding class implements the dataset preparation principle by providing a callable that processes lists of tokenized samples into properly padded and batched tensors. It receives individual samples from the dataset loader and dynamically pads them to create uniform batches suitable for model input.

This implementation leverages the tokenizer's padding capabilities to handle various padding strategies, automatically generates attention masks, and ensures proper tensor formatting for PyTorch or other frameworks. It intelligently renames label fields to match model expectations and handles edge cases like single-sample batches.

=== Usage ===
Import and instantiate DataCollatorWithPadding after creating your tokenizer but before initializing the Trainer. Pass your tokenizer instance and optionally specify padding behavior. The Trainer will automatically use this collator to prepare batches during training and evaluation. Use it for any text classification, sequence labeling, or language modeling task where tokenized text needs to be batched efficiently.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/data/data_collator.py:L190-240

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.0 (Volta).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", or "pt".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: bool | str | PaddingStrategy = True
    max_length: int | None = None
    pad_to_multiple_of: int | None = None
    return_tensors: str = "pt"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Process a list of samples into a padded batch."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import DataCollatorWithPadding
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| tokenizer || PreTrainedTokenizerBase || Yes || Tokenizer instance used to encode the data, provides padding configuration
|-
| padding || bool or str || No || Padding strategy: True/"longest" (batch-wise), "max_length", or False (default: True)
|-
| max_length || int || No || Maximum sequence length for padding when using "max_length" strategy (default: None)
|-
| pad_to_multiple_of || int || No || Pad sequences to multiple of this value for Tensor Core optimization (default: None)
|-
| return_tensors || str || No || Tensor type to return: "pt" for PyTorch, "np" for NumPy (default: "pt")
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| data_collator || DataCollatorWithPadding || Callable object that processes batches, used by Trainer's DataLoader
|}

== Usage Examples ==

=== Basic Text Classification ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
from datasets import load_dataset

# Load tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = load_dataset("glue", "mrpc", split="train")

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Create data collator with dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Use with Trainer - batches will be automatically padded
training_args = TrainingArguments(output_dir="./results")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)
</syntaxhighlight>

=== Optimized Padding for GPU Training ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer, DataCollatorWithPadding

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Pad to multiple of 8 for Tensor Core optimization on modern GPUs
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,  # Dynamic padding to longest in batch
    pad_to_multiple_of=8,  # Optimize for Tensor Cores
    return_tensors="pt"
)

# Example: Manual batch creation to see the collator in action
samples = [
    {"input_ids": [101, 2023, 2003, 102]},  # Length 4
    {"input_ids": [101, 1037, 6263, 6251, 102]},  # Length 5
]

# Collator pads to length 8 (next multiple of 8 from max length 5)
batch = data_collator(samples)
print(batch["input_ids"].shape)  # torch.Size([2, 8])
print(batch["attention_mask"])   # Shows which tokens are real vs padding
</syntaxhighlight>

=== Fixed-Length Padding ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer, DataCollatorWithPadding

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Pad all sequences to fixed length (useful for consistent batch sizes)
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding="max_length",
    max_length=128,  # Always pad/truncate to 128 tokens
    return_tensors="pt"
)

# All batches will have shape [batch_size, 128] regardless of input lengths
</syntaxhighlight>

=== Using with DataLoader ===
<syntaxhighlight lang="python">
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = load_dataset("imdb", split="train")

# Tokenize
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Create collator and dataloader
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = DataLoader(
    tokenized_dataset,
    batch_size=16,
    collate_fn=data_collator,
    shuffle=True
)

# Batches are automatically padded during iteration
for batch in train_dataloader:
    # batch["input_ids"] has shape [16, max_length_in_batch]
    # batch["attention_mask"] indicates real vs padded tokens
    outputs = model(**batch)
</syntaxhighlight>

=== Custom Collator with Label Processing ===
<syntaxhighlight lang="python">
from transformers import DataCollatorWithPadding, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# The collator automatically handles label field renaming
# If your data has "label" field, it gets converted to "labels" (model convention)
samples = [
    {"input_ids": [101, 2023, 102], "label": 1},
    {"input_ids": [101, 2008, 2003, 102], "label": 0},
]

batch = data_collator(samples)
# batch now has "labels" key instead of "label" to match model expectations
assert "labels" in batch
assert "label" not in batch
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Dataset_Preparation]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Training_Environment]]
