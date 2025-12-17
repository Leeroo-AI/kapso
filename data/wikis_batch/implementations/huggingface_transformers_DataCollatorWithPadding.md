{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Training]], [[domain::Data Processing]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Concrete data collator class for dynamically padding inputs within training batches provided by the HuggingFace Transformers library.

=== Description ===

DataCollatorWithPadding is a callable dataclass that dynamically pads sequences in a batch to the same length. It receives a list of tokenized examples and pads them according to the tokenizer's padding configuration, ensuring all sequences in a batch have the same length. This is crucial for efficient batch processing on GPUs. The collator handles padding direction, padding tokens, and automatically converts labels from "label" or "label_ids" fields to "labels" as expected by model forward methods.

=== Usage ===

Use DataCollatorWithPadding when training or evaluating transformer models with variable-length sequences. Dynamic padding (padding to the longest sequence in each batch) is more memory-efficient than padding all sequences to a fixed maximum length during tokenization. This is the recommended approach for most training scenarios, as it balances efficiency with memory usage. Pass it to the Trainer's data_collator parameter.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/data/data_collator.py
* '''Lines:''' 190-240

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class DataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: bool | str | PaddingStrategy = True
    max_length: int | None = None
    pad_to_multiple_of: int | None = None
    return_tensors: str = "pt"

    def __call__(
        self,
        features: list[dict[str, Any]]
    ) -> dict[str, Any]:
        ...
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
| tokenizer || PreTrainedTokenizerBase || Yes || The tokenizer used for encoding the data
|-
| padding || bool, str, or PaddingStrategy || No (default: True) || Padding strategy: True/'longest' (pad to longest in batch), 'max_length' (pad to max_length), False/'do_not_pad' (no padding)
|-
| max_length || int || No || Maximum length for padding (only used with padding='max_length')
|-
| pad_to_multiple_of || int || No || Pad to a multiple of this value (useful for Tensor Cores on NVIDIA GPUs with compute capability >= 7.0)
|-
| return_tensors || str || No (default: "pt") || Type of tensors to return: "pt" (PyTorch), "tf" (TensorFlow), or "np" (NumPy)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| batch || dict[str, Any] || Dictionary containing padded tensors for input_ids, attention_mask, and labels (if present)
|}

== Usage Examples ==

=== Basic Usage ===
<syntaxhighlight lang="python">
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize dataset without padding (padding will be done dynamically)
dataset = load_dataset("imdb", split="train[:100]")

def tokenize_function(examples):
    # No padding here - let the data collator handle it
    return tokenizer(examples["text"], truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Create data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Use with Trainer
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    num_train_epochs=3
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator  # Dynamic padding during training
)

trainer.train()
</syntaxhighlight>

=== Advanced Configuration ===
<syntaxhighlight lang="python">
from transformers import AutoTokenizer, DataCollatorWithPadding

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Pad to multiple of 8 for Tensor Core optimization
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,  # Pad to longest sequence in batch
    pad_to_multiple_of=8,  # Optimize for Tensor Cores
    return_tensors="pt"
)

# Example usage
features = [
    {"input_ids": [101, 2054, 2003, 102], "attention_mask": [1, 1, 1, 1]},
    {"input_ids": [101, 2023, 2003, 1037, 2937, 6251, 102], "attention_mask": [1, 1, 1, 1, 1, 1, 1]}
]

batch = data_collator(features)
# batch["input_ids"].shape will be [2, 8] (padded to multiple of 8)
# batch["attention_mask"].shape will be [2, 8]

# Fixed padding to max_length
data_collator_fixed = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding="max_length",
    max_length=512
)
# All batches will be padded to 512 tokens
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Data_Collation]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_PyTorch]]
