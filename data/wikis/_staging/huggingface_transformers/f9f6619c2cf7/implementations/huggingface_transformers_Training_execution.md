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
Concrete training execution method that orchestrates the complete training process provided by HuggingFace Transformers Trainer.

=== Description ===
The train() method implements the training loop principle by providing a comprehensive, production-ready training execution engine. It manages the complete training lifecycle from initialization through completion, handling epoch iteration, batch processing, gradient computation and application, periodic evaluation, checkpoint saving, logging, and proper cleanup.

This implementation abstracts away the complexity of distributed training, mixed precision, gradient accumulation, and device management while providing hooks for customization through callbacks. It automatically handles training resumption, hyperparameter search integration, and proper state management for reproducibility. The method returns detailed training output including final loss, metrics history, and training statistics.

=== Usage ===
Call train() on an initialized Trainer instance after all components (model, args, datasets) have been configured. This is the main entry point for training execution. Optionally pass resume_from_checkpoint to continue interrupted training, trial for hyperparameter optimization, or ignore_keys_for_eval for custom evaluation handling. The method blocks until training completes or is interrupted, returning a TrainOutput object with results.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/trainer.py:L2068-2173

=== Signature ===
<syntaxhighlight lang="python">
def train(
    self,
    resume_from_checkpoint: str | bool | None = None,
    trial: Union["optuna.Trial", dict[str, Any], None] = None,
    ignore_keys_for_eval: list[str] | None = None,
):
    """
    Main training entry point.

    Args:
        resume_from_checkpoint (`str` or `bool`, *optional*):
            If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
            `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
            of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
        trial (`optuna.Trial` or `dict[str, Any]`, *optional*):
            The trial run or the hyperparameter dictionary for hyperparameter search.
        ignore_keys_for_eval (`list[str]`, *optional*)
            A list of keys in the output of your model (if it is a dictionary) that should be ignored when
            gathering predictions for evaluation during the training.

    Returns:
        TrainOutput: Object containing training metrics and final loss.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import Trainer
# train() is a method of Trainer class
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| resume_from_checkpoint || str or bool || No || Path to checkpoint directory, or True to auto-detect last checkpoint
|-
| trial || optuna.Trial or dict || No || Trial object for hyperparameter search integration
|-
| ignore_keys_for_eval || list[str] || No || Model output keys to exclude from evaluation predictions
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| train_output || TrainOutput || Contains global_step, training_loss, metrics, and other training statistics
|}

== Usage Examples ==

=== Basic Training ===
<syntaxhighlight lang="python">
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset

# Setup model and data
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

dataset = load_dataset("glue", "mrpc")
def tokenize(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)
tokenized_dataset = dataset.map(tokenize, batched=True)

# Configure training
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    logging_steps=100,
    eval_strategy="epoch"
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=DataCollatorWithPadding(tokenizer)
)

# Execute training
train_output = trainer.train()

# Access training results
print(f"Final loss: {train_output.training_loss}")
print(f"Total steps: {train_output.global_step}")
print(f"Training metrics: {train_output.metrics}")
</syntaxhighlight>

=== Training with Checkpointing and Resumption ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")

# Enable checkpointing
training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,  # Keep only 3 most recent checkpoints
    logging_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Initial training (may be interrupted)
try:
    train_output = trainer.train()
except KeyboardInterrupt:
    print("Training interrupted")

# Resume from last checkpoint
# Option 1: Automatic detection
trainer_resumed = Trainer(model=model, args=training_args, train_dataset=train_dataset)
train_output = trainer_resumed.train(resume_from_checkpoint=True)

# Option 2: Explicit checkpoint path
train_output = trainer_resumed.train(resume_from_checkpoint="./checkpoints/checkpoint-1000")

print(f"Training completed at step {train_output.global_step}")
</syntaxhighlight>

=== Training with Evaluation and Best Model Tracking ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from datasets import load_metric
import numpy as np

model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=3)

# Define metric computation
accuracy = load_metric("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# Configure to track best model
training_args = TrainingArguments(
    output_dir="./best-model-training",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    load_best_model_at_end=True,  # Load best checkpoint at end
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_steps=50
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Train - best model automatically loaded at end
train_output = trainer.train()

# Model is now the best checkpoint, not the final one
print(f"Best model metric: {train_output.metrics.get('best_metric')}")
</syntaxhighlight>

=== Training with Progress Monitoring ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments, TrainerCallback

# Custom callback to monitor training
class ProgressCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f"\n=== Starting Epoch {state.epoch} ===")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:
            lr = state.log_history[-1].get("learning_rate", "N/A")
            loss = state.log_history[-1].get("loss", "N/A")
            print(f"Step {state.global_step}: loss={loss:.4f}, lr={lr:.2e}")

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Epoch {state.epoch} completed")

training_args = TrainingArguments(
    output_dir="./monitored-training",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    logging_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[ProgressCallback()]
)

train_output = trainer.train()
</syntaxhighlight>

=== Training with Gradient Accumulation ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments, AutoModelForSeq2SeqLM

# Large model that doesn't fit with large batch size
model = AutoModelForSeq2SeqLM.from_pretrained("t5-large")

training_args = TrainingArguments(
    output_dir="./gradient-accumulation",
    num_train_epochs=3,
    per_device_train_batch_size=4,  # Small batch per device
    gradient_accumulation_steps=8,   # Accumulate over 8 batches
    # Effective batch size = 4 * 8 = 32
    learning_rate=3e-5,
    logging_steps=50,
    fp16=True,  # Mixed precision for memory efficiency
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Gradients accumulated over 8 steps before each optimizer update
train_output = trainer.train()
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
</syntaxhighlight>

=== Training with Early Stopping ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import numpy as np

training_args = TrainingArguments(
    output_dir="./early-stopping",
    num_train_epochs=20,  # Max epochs
    per_device_train_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

# Early stopping: stop if eval_loss doesn't improve for 3 epochs
early_stop_callback = EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.0
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[early_stop_callback]
)

# Training may stop before reaching 20 epochs
train_output = trainer.train()
print(f"Training stopped at epoch {train_output.metrics.get('epoch')}")
</syntaxhighlight>

=== Distributed Training ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM

# Launch with: torchrun --nproc_per_node=4 train_script.py
# Or use accelerate launch

model = AutoModelForCausalLM.from_pretrained("gpt2-medium")

training_args = TrainingArguments(
    output_dir="./distributed-training",
    num_train_epochs=3,
    per_device_train_batch_size=8,  # Per GPU
    # Global batch size = 8 * 4 GPUs = 32
    learning_rate=5e-5,
    logging_steps=50,
    ddp_backend="nccl",  # Distributed backend
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Automatically handles distributed training
train_output = trainer.train()

# Only rank 0 prints final results
if training_args.local_rank in [-1, 0]:
    print(f"Distributed training completed: {train_output.global_step} steps")
</syntaxhighlight>

=== Hyperparameter Search Integration ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification

def model_init():
    """Return fresh model instance for each trial."""
    return AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./hp-search",
    eval_strategy="epoch"
)

trainer = Trainer(
    model_init=model_init,  # Function, not instance
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Run hyperparameter search
best_trial = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    n_trials=10
)

# Train with best hyperparameters
# The train() method is called internally for each trial
trainer.train()
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Training_Loop]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Training_Environment]]
