{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Datasets Library|https://huggingface.co/docs/datasets]]
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Training]], [[domain::Data Processing]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Wrapper documentation for dataset tokenization using the HuggingFace datasets library's map function with transformers tokenizers.

=== Description ===

Dataset tokenization is performed using the datasets library's map() method in conjunction with transformers tokenizers. The map() function applies a tokenization function to every example in the dataset, converting raw text into token IDs, attention masks, and other inputs required by transformer models. This process is optimized for performance through batching and multiprocessing, and can automatically remove unnecessary columns.

While the datasets library is external to transformers, this is the standard and recommended approach for preparing data for transformer model training in the HuggingFace ecosystem.

=== Usage ===

Use dataset.map() with a tokenization function when you need to convert text data into tokenized inputs for transformer models. This is essential before training or fine-tuning, as models require numerical token IDs rather than raw text. The batched parameter enables efficient vectorized tokenization, and remove_columns helps clean up the dataset by removing raw text fields after tokenization.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/datasets datasets]
* '''Library:''' External dependency (datasets library)
* '''Integration:''' Works seamlessly with transformers tokenizers

=== Signature ===
<syntaxhighlight lang="python">
Dataset.map(
    function: Callable | None = None,
    with_indices: bool = False,
    with_rank: bool = False,
    input_columns: str | list[str] | None = None,
    batched: bool = False,
    batch_size: int | None = 1000,
    drop_last_batch: bool = False,
    remove_columns: str | list[str] | None = None,
    keep_in_memory: bool = False,
    load_from_cache_file: bool | None = None,
    cache_file_name: str | None = None,
    writer_batch_size: int | None = 1000,
    features: Features | None = None,
    disable_nullable: bool = False,
    fn_kwargs: dict | None = None,
    num_proc: int | None = None,
    suffix_template: str = "_{rank:05d}_of_{num_proc:05d}",
    new_fingerprint: str | None = None,
    desc: str | None = None
) -> Dataset
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from datasets import load_dataset
from transformers import AutoTokenizer
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| function || Callable || Yes || The tokenization function to apply to each example or batch
|-
| batched || bool || No (default: False) || Whether to process examples in batches for efficiency
|-
| batch_size || int || No (default: 1000) || Number of examples per batch when batched=True
|-
| remove_columns || str or list[str] || No || Column names to remove from the dataset after mapping
|-
| num_proc || int || No || Number of processes for multiprocessing
|-
| fn_kwargs || dict || No || Additional keyword arguments to pass to the function
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| tokenized_dataset || Dataset || Dataset with added tokenized fields (input_ids, attention_mask, etc.) and optionally removed raw text columns
|}

== Usage Examples ==

=== Basic Usage ===
<syntaxhighlight lang="python">
from datasets import load_dataset
from transformers import AutoTokenizer

# Load dataset and tokenizer
dataset = load_dataset("imdb", split="train")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Define tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

# Apply tokenization with batching
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# The tokenized_dataset now contains:
# - input_ids: token IDs
# - attention_mask: attention mask for padding
# - token_type_ids: segment IDs (if applicable)
# Original "text" column is removed

# Advanced usage with multiprocessing and custom parameters
def tokenize_and_label(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    # Add labels (assuming binary classification)
    tokenized["labels"] = examples["label"]
    return tokenized

tokenized_dataset = dataset.map(
    tokenize_and_label,
    batched=True,
    batch_size=1000,
    num_proc=4,  # Use 4 processes for faster processing
    remove_columns=["text"],
    desc="Tokenizing dataset"
)
</syntaxhighlight>

=== Integration with Transformers Training ===
<syntaxhighlight lang="python">
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# Load and tokenize dataset
dataset = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["sentence1", "sentence2", "idx"]
)

# Ready to use with Trainer
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"]
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Dataset_Preparation]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_PyTorch]]
* [[requires_lib::Library:datasets]]
