{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Trainer Documentation|https://huggingface.co/docs/transformers/main_classes/trainer]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Training]], [[domain::Evaluation]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Concrete evaluation execution method that assesses model performance on validation data provided by HuggingFace Transformers Trainer.

=== Description ===
The evaluate() method implements the evaluation loop principle by providing a complete evaluation execution engine that systematically measures model performance on held-out data. It handles the entire evaluation workflow: setting up evaluation mode, iterating through evaluation data, collecting predictions, computing metrics via user-provided functions, and returning comprehensive results.

This implementation automatically handles distributed evaluation across multiple devices, memory-efficient batch processing, multiple dataset evaluation, and integration with the training lifecycle. It ensures proper model state management, gradient disabling, and metric aggregation. The method can be called standalone or is automatically invoked during training based on the evaluation strategy.

=== Usage ===
Call evaluate() on an initialized Trainer instance to assess model performance. Can be called at any time - during training (automatically per eval_strategy), after training completes, or on a standalone model without training. Optionally pass eval_dataset to evaluate on different data than the default, ignore_keys to exclude certain model outputs, or metric_key_prefix to customize metric names. Returns a dictionary containing loss and all computed metrics.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/trainer.py:L4228-4327

=== Signature ===
<syntaxhighlight lang="python">
def evaluate(
    self,
    eval_dataset: Dataset | dict[str, Dataset] | None = None,
    ignore_keys: list[str] | None = None,
    metric_key_prefix: str = "eval",
) -> dict[str, float]:
    """
    Run evaluation and returns metrics.

    The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
    (pass it to the init `compute_metrics` argument).

    You can also subclass and override this method to inject custom behavior.

    Args:
        eval_dataset (Union[`Dataset`, dict[str, `Dataset`]], *optional*):
            Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
            not accepted by the `model.forward()` method are automatically removed. If it is a dictionary, it will
            evaluate on each dataset, prepending the dictionary key to the metric name. Datasets must implement the
            `__len__` method.

        ignore_keys (`list[str]`, *optional*):
            A list of keys in the output of your model (if it is a dictionary) that should be ignored when
            gathering predictions.

        metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
            An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
            "eval_bleu" if the prefix is "eval" (default)

    Returns:
        A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
        dictionary also contains the epoch number which comes from the training state.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import Trainer
# evaluate() is a method of Trainer class
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| eval_dataset || Dataset or dict[str, Dataset] || No || Dataset(s) to evaluate on (defaults to self.eval_dataset from initialization)
|-
| ignore_keys || list[str] || No || Model output dictionary keys to exclude from prediction gathering
|-
| metric_key_prefix || str || No || Prefix for metric names in returned dictionary (default: "eval")
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| metrics || dict[str, float] || Dictionary with loss, computed metrics, epoch number, and evaluation statistics
|}

== Usage Examples ==

=== Basic Evaluation ===
<syntaxhighlight lang="python">
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset, load_metric
import numpy as np

# Setup
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

dataset = load_dataset("glue", "mrpc")
def tokenize(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)
tokenized_dataset = dataset.map(tokenize, batched=True)

# Define metrics
accuracy_metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Initialize trainer
training_args = TrainingArguments(output_dir="./results")
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_dataset["validation"],
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

# Evaluate (can call without training)
eval_results = trainer.evaluate()

print(f"Evaluation loss: {eval_results['eval_loss']:.4f}")
print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"Samples evaluated: {eval_results['eval_samples']}")
</syntaxhighlight>

=== Evaluation During Training ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=3)

# Configure automatic evaluation during training
training_args = TrainingArguments(
    output_dir="./training-with-eval",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    eval_strategy="steps",  # Evaluate every N steps
    eval_steps=500,         # Evaluate every 500 steps
    logging_steps=100,
    save_strategy="steps",
    save_steps=500
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# evaluate() is called automatically every 500 steps during training
train_output = trainer.train()

# Can also call manually after training
final_metrics = trainer.evaluate()
print("Final evaluation metrics:", final_metrics)
</syntaxhighlight>

=== Multiple Evaluation Datasets ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(output_dir="./multi-eval")

# Evaluate on multiple datasets
eval_datasets = {
    "validation": validation_dataset,
    "test_easy": test_easy_dataset,
    "test_hard": test_hard_dataset
}

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=eval_datasets,  # Dictionary of datasets
    compute_metrics=compute_metrics
)

# Evaluates on all datasets, metrics prefixed with dataset names
eval_results = trainer.evaluate()

print(f"Validation accuracy: {eval_results['eval_validation_accuracy']:.4f}")
print(f"Test easy accuracy: {eval_results['eval_test_easy_accuracy']:.4f}")
print(f"Test hard accuracy: {eval_results['eval_test_hard_accuracy']:.4f}")
</syntaxhighlight>

=== Custom Evaluation Dataset ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(output_dir="./custom-eval")

# Initialize with training dataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    compute_metrics=compute_metrics
)

# Train model
trainer.train()

# Evaluate on different datasets by passing them explicitly
val_metrics = trainer.evaluate(eval_dataset=validation_dataset)
test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
ood_metrics = trainer.evaluate(eval_dataset=out_of_domain_dataset, metric_key_prefix="ood")

print("Validation:", val_metrics)
print("Test:", test_metrics)
print("Out-of-domain:", ood_metrics)
</syntaxhighlight>

=== Comprehensive Metrics Computation ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments
from datasets import load_metric
import numpy as np

# Load multiple metrics
accuracy_metric = load_metric("accuracy")
precision_metric = load_metric("precision")
recall_metric = load_metric("recall")
f1_metric = load_metric("f1")

def compute_comprehensive_metrics(eval_pred):
    """Compute multiple evaluation metrics."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "precision": precision_metric.compute(predictions=predictions, references=labels, average="weighted")["precision"],
        "recall": recall_metric.compute(predictions=predictions, references=labels, average="weighted")["recall"],
        "f1": f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"],
    }

training_args = TrainingArguments(
    output_dir="./comprehensive-eval",
    eval_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_comprehensive_metrics
)

# Evaluation returns all computed metrics
eval_results = trainer.evaluate()
print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"Precision: {eval_results['eval_precision']:.4f}")
print(f"Recall: {eval_results['eval_recall']:.4f}")
print(f"F1: {eval_results['eval_f1']:.4f}")
</syntaxhighlight>

=== Evaluation with Best Model Selection ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./best-model",
    num_train_epochs=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,      # Load best checkpoint at end
    metric_for_best_model="eval_f1",  # Use F1 for selection
    greater_is_better=True            # Higher F1 is better
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Train with periodic evaluation
trainer.train()

# Trainer automatically loaded best checkpoint based on eval_f1
# Now evaluate final model (which is the best checkpoint)
final_eval = trainer.evaluate()
print(f"Best model F1 score: {final_eval['eval_f1']:.4f}")
</syntaxhighlight>

=== Standalone Evaluation Without Training ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification

# Load a pre-trained or fine-tuned model
model = AutoModelForSequenceClassification.from_pretrained("./my-fine-tuned-model")

training_args = TrainingArguments(output_dir="./eval-only")

# Create trainer just for evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Evaluate without any training
test_results = trainer.evaluate()
print("Test set results:", test_results)
</syntaxhighlight>

=== Evaluation with Custom Metric Prefix ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
training_args = TrainingArguments(output_dir="./prefix-eval")

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics
)

# Different prefixes for different evaluation contexts
val_results = trainer.evaluate(
    eval_dataset=validation_dataset,
    metric_key_prefix="validation"
)

test_results = trainer.evaluate(
    eval_dataset=test_dataset,
    metric_key_prefix="test"
)

# Metrics have different prefixes
print(f"Validation loss: {val_results['validation_loss']}")
print(f"Test loss: {test_results['test_loss']}")
</syntaxhighlight>

=== Generation Task Evaluation ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_metric
import numpy as np

model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
tokenizer = AutoTokenizer.from_pretrained("t5-small")

# BLEU metric for generation
bleu_metric = load_metric("bleu")

def compute_generation_metrics(eval_pred):
    """Compute BLEU score for generation."""
    predictions, labels = eval_pred

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in labels (used for padding)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute BLEU
    result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)

    return {"bleu": result["bleu"]}

training_args = TrainingArguments(
    output_dir="./generation-eval",
    predict_with_generate=True,  # Generate sequences during evaluation
    eval_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_generation_metrics,
    processing_class=tokenizer
)

eval_results = trainer.evaluate()
print(f"BLEU score: {eval_results['eval_bleu']:.4f}")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Evaluation_Loop]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Training_Environment]]
