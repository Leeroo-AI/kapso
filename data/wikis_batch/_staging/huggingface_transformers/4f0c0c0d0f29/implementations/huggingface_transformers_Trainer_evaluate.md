{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Training]], [[domain::Evaluation]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Concrete evaluation method for the Trainer class provided by the HuggingFace Transformers library.

=== Description ===

The Trainer.evaluate() method runs evaluation on a dataset and returns computed metrics. It creates an evaluation dataloader, runs the evaluation loop with the model in eval mode, computes predictions, calculates metrics using the compute_metrics function (if provided), and returns a dictionary of results. The method handles multiple evaluation datasets, memory tracking, speed metrics calculation, and callback notifications. It supports both standard and distributed evaluation scenarios.

=== Usage ===

Use trainer.evaluate() to assess model performance on validation or test datasets. Call it during or after training to measure metrics like accuracy, loss, F1 score, etc. The method can be called standalone for model evaluation without training, or automatically during training when eval_strategy is configured in TrainingArguments. It's essential for model selection, hyperparameter tuning, and performance monitoring.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/trainer.py
* '''Lines:''' 4228-4327

=== Signature ===
<syntaxhighlight lang="python">
def evaluate(
    self,
    eval_dataset: Dataset | dict[str, Dataset] | None = None,
    ignore_keys: list[str] | None = None,
    metric_key_prefix: str = "eval",
) -> dict[str, float]
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import Trainer
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| eval_dataset || Dataset or dict[str, Dataset] || No (uses self.eval_dataset) || Dataset to evaluate on. If dict, evaluates on each dataset separately with keys as name prefixes
|-
| ignore_keys || list[str] || No || Keys in model output to ignore when gathering predictions
|-
| metric_key_prefix || str || No (default: "eval") || Prefix for metric names in the returned dictionary
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| metrics || dict[str, float] || Dictionary containing evaluation metrics (loss, custom metrics, speed metrics, epoch number)
|}

== Usage Examples ==

=== Basic Usage ===
<syntaxhighlight lang="python">
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score

# Setup
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Prepare dataset
dataset = load_dataset("imdb", split="test[:1000]")
tokenized_dataset = dataset.map(
    lambda x: tokenizer(x["text"], truncation=True, max_length=512),
    batched=True
)

# Define metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Create trainer
training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=64
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_dataset,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)

# Run evaluation
eval_results = trainer.evaluate()

print(f"Evaluation loss: {eval_results['eval_loss']}")
print(f"Accuracy: {eval_results['eval_accuracy']}")
print(f"Samples per second: {eval_results['eval_samples_per_second']}")
</syntaxhighlight>

=== Evaluation During Training ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments

# Configure automatic evaluation during training
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    eval_strategy="epoch",  # Evaluate at end of each epoch
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train with automatic evaluation
train_result = trainer.train()

# Final evaluation
final_eval = trainer.evaluate()
print(f"Final accuracy: {final_eval['eval_accuracy']}")
</syntaxhighlight>

=== Multiple Dataset Evaluation ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Load multiple evaluation datasets
eval_dataset_1 = load_dataset("imdb", split="test[:500]")
eval_dataset_2 = load_dataset("sst2", split="validation")

# Tokenize both
def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

eval_1_tokenized = eval_dataset_1.map(tokenize, batched=True)
eval_2_tokenized = eval_dataset_2.map(tokenize, batched=True)

# Create trainer
trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="./results"),
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)

# Evaluate on multiple datasets
eval_results = trainer.evaluate(
    eval_dataset={
        "imdb": eval_1_tokenized,
        "sst2": eval_2_tokenized
    }
)

# Results have prefixes for each dataset
print(f"IMDB accuracy: {eval_results['eval_imdb_accuracy']}")
print(f"SST2 accuracy: {eval_results['eval_sst2_accuracy']}")
print(f"IMDB loss: {eval_results['eval_imdb_loss']}")
print(f"SST2 loss: {eval_results['eval_sst2_loss']}")
</syntaxhighlight>

=== Evaluation with Custom Prefix ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="./results"),
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

# Evaluate with custom prefix for test set
test_results = trainer.evaluate(
    eval_dataset=test_dataset,
    metric_key_prefix="test"
)

print(f"Test accuracy: {test_results['test_accuracy']}")
print(f"Test loss: {test_results['test_loss']}")

# Evaluate validation set with different prefix
val_results = trainer.evaluate(
    eval_dataset=validation_dataset,
    metric_key_prefix="validation"
)

print(f"Validation accuracy: {val_results['validation_accuracy']}")
</syntaxhighlight>

=== Standalone Evaluation (No Training) ===
<syntaxhighlight lang="python">
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

# Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(
    "textattack/bert-base-uncased-imdb"
)
tokenizer = AutoTokenizer.from_pretrained(
    "textattack/bert-base-uncased-imdb"
)

# Prepare test data
test_dataset = load_dataset("imdb", split="test")
test_tokenized = test_dataset.map(
    lambda x: tokenizer(x["text"], truncation=True),
    batched=True
)

# Create trainer just for evaluation
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./temp",
        per_device_eval_batch_size=64
    ),
    compute_metrics=compute_metrics
)

# Evaluate without training
results = trainer.evaluate(eval_dataset=test_tokenized)
print(results)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Evaluation_Checkpointing]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_PyTorch]]
