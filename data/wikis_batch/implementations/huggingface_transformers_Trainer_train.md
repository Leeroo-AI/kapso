{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Concrete training loop execution method for the Trainer class provided by the HuggingFace Transformers library.

=== Description ===

The Trainer.train() method is the main entry point for executing the training loop. It handles the complete training process including checkpoint resumption, model initialization, data loading, forward/backward passes, optimizer steps, gradient accumulation, mixed precision training, distributed training synchronization, logging, evaluation, and checkpointing. The method orchestrates all training components configured during Trainer initialization and returns training metrics and final model state.

=== Usage ===

Use trainer.train() to execute the training loop after initializing a Trainer instance. Call it to begin training your model according to the TrainingArguments configuration. The method handles all complexity of modern deep learning training including automatic batch size finding, gradient accumulation, evaluation scheduling, checkpoint saving, and distributed training coordination. Optionally pass a checkpoint path to resume interrupted training.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/trainer.py
* '''Lines:''' 2068-2170

=== Signature ===
<syntaxhighlight lang="python">
def train(
    self,
    resume_from_checkpoint: str | bool | None = None,
    trial: Union["optuna.Trial", dict[str, Any], None] = None,
    ignore_keys_for_eval: list[str] | None = None,
) -> TrainOutput
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
| resume_from_checkpoint || str or bool || No || If str: path to checkpoint directory. If True: load last checkpoint from output_dir. If None/False: start training from scratch
|-
| trial || optuna.Trial or dict || No || The trial run or hyperparameter dictionary for hyperparameter search
|-
| ignore_keys_for_eval || list[str] || No || Keys in model output dictionary to ignore when gathering predictions for evaluation
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| train_output || TrainOutput || Named tuple containing global_step (int), training_loss (float), and metrics (dict) from the training run
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

# Setup model, tokenizer, and data
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

dataset = load_dataset("imdb", split="train[:1000]")
tokenized_dataset = dataset.map(
    lambda x: tokenizer(x["text"], truncation=True),
    batched=True
)

# Configure training
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    logging_steps=10,
    save_strategy="epoch"
)

# Create and run trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)

# Execute training
train_result = trainer.train()

# Access results
print(f"Training completed in {train_result.metrics['train_runtime']} seconds")
print(f"Final training loss: {train_result.training_loss}")
print(f"Total steps: {train_result.global_step}")
</syntaxhighlight>

=== Training with Evaluation ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score

# Define metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Split dataset
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)

# Configure training with evaluation
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    eval_strategy="epoch",  # Evaluate after each epoch
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=50
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_test_split["train"],
    eval_dataset=train_test_split["test"],
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)

# Train with automatic evaluation
train_result = trainer.train()

# Results include evaluation metrics
print(f"Best model accuracy: {trainer.state.best_metric}")
</syntaxhighlight>

=== Resuming from Checkpoint ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments

# Initial training
training_args = TrainingArguments(
    output_dir="./my_model",
    num_train_epochs=5,
    save_strategy="epoch",
    save_total_limit=2  # Keep only last 2 checkpoints
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

try:
    # Start training
    trainer.train()
except KeyboardInterrupt:
    print("Training interrupted!")

# Resume from last checkpoint
print("Resuming training from last checkpoint...")
trainer.train(resume_from_checkpoint=True)

# Or resume from specific checkpoint
trainer.train(resume_from_checkpoint="./my_model/checkpoint-500")
</syntaxhighlight>

=== Advanced Training with Mixed Precision and Gradient Accumulation ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments

# Advanced configuration
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,  # Smaller batch per device
    gradient_accumulation_steps=4,   # Effective batch size = 8 * 4 = 32
    fp16=True,                        # Enable mixed precision (if GPU supports it)
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="tensorboard"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train with all optimizations
train_result = trainer.train()

# Save final metrics
trainer.save_metrics("train", train_result.metrics)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Training_Loop]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_PyTorch]]
