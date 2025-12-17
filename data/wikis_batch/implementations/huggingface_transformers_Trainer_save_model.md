{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Training]], [[domain::Model Serialization]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Concrete model saving method for the Trainer class provided by the HuggingFace Transformers library.

=== Description ===

The Trainer.save_model() method saves the trained model to disk so it can be reloaded using from_pretrained(). It handles various distributed training scenarios including TPU, SageMaker Model Parallelism, FSDP, DeepSpeed, and standard single/multi-GPU training. The method saves model weights, configuration, and tokenizer (if available) to the specified directory. It only saves from the main process in distributed settings and handles special cases like model parallelism, quantization, and state dict consolidation.

=== Usage ===

Use trainer.save_model() to persist a trained model to disk after training or at specific checkpoints. The saved model can be loaded later using AutoModel.from_pretrained() or similar methods. This is essential for model deployment, sharing, checkpointing, and continuing training later. The method is automatically called during training if save_strategy is configured, but can also be called manually at any time.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/trainer.py
* '''Lines:''' 3989-4053

=== Signature ===
<syntaxhighlight lang="python">
def save_model(
    self,
    output_dir: str | None = None,
    _internal_call: bool = False
) -> None
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
| output_dir || str || No (uses self.args.output_dir) || The directory where the model will be saved
|-
| _internal_call || bool || No (default: False) || Internal flag to indicate if called by Trainer (affects push_to_hub behavior)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| saved_files || None (side effect) || Creates directory with model files: pytorch_model.bin or model.safetensors, config.json, tokenizer files, training_args.bin
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

# Setup and train model
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

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)

# Train
trainer.train()

# Save the final model
trainer.save_model("./my_fine_tuned_model")

# Model can now be loaded
loaded_model = AutoModelForSequenceClassification.from_pretrained(
    "./my_fine_tuned_model"
)
</syntaxhighlight>

=== Automatic Saving During Training ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments

# Configure automatic model saving
training_args = TrainingArguments(
    output_dir="./model_checkpoints",
    num_train_epochs=5,
    save_strategy="epoch",        # Save after each epoch
    save_total_limit=3,            # Keep only last 3 checkpoints
    load_best_model_at_end=True,  # Load best model when training ends
    metric_for_best_model="accuracy",
    eval_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Training automatically saves checkpoints
trainer.train()

# Best model is automatically loaded at the end
# Save it to a final location
trainer.save_model("./final_model")
</syntaxhighlight>

=== Save at Specific Steps ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./model_checkpoints",
    num_train_epochs=3,
    save_strategy="steps",    # Save every N steps
    save_steps=500,           # Save every 500 steps
    save_total_limit=5,       # Keep only last 5 checkpoints
    logging_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()

# Checkpoints saved at: checkpoint-500, checkpoint-1000, checkpoint-1500, etc.
# Final model saved in output_dir
</syntaxhighlight>

=== Manual Checkpointing ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments
import os

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    save_strategy="no"  # Disable automatic saving
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Custom training loop with manual checkpoints
for epoch in range(10):
    # Train for one epoch
    trainer.train()

    # Manually save after each epoch
    checkpoint_dir = f"./manual_checkpoints/epoch_{epoch+1}"
    trainer.save_model(checkpoint_dir)
    print(f"Saved checkpoint to {checkpoint_dir}")

    # Optional: evaluate
    metrics = trainer.evaluate()
    print(f"Epoch {epoch+1} metrics: {metrics}")
</syntaxhighlight>

=== Saving with Tokenizer and Custom Components ===
<syntaxhighlight lang="python">
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification
)

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

training_args = TrainingArguments(output_dir="./results")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer  # Pass tokenizer to trainer
)

trainer.train()

# Save model - this saves both model and tokenizer
trainer.save_model("./complete_model")

# Directory contains:
# - config.json (model configuration)
# - pytorch_model.bin or model.safetensors (model weights)
# - tokenizer_config.json (tokenizer configuration)
# - vocab.txt (vocabulary)
# - special_tokens_map.json (special tokens)
# - training_args.bin (training arguments)

# Reload everything
loaded_model = AutoModelForSequenceClassification.from_pretrained(
    "./complete_model"
)
loaded_tokenizer = AutoTokenizer.from_pretrained("./complete_model")
</syntaxhighlight>

=== Integration with Hub ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments

# Configure to push to HuggingFace Hub
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    push_to_hub=True,
    hub_model_id="my-username/my-awesome-model",
    hub_strategy="every_save"  # Push every time model is saved
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()

# Save and automatically push to hub
trainer.save_model("./final_model")
# Model is now available at huggingface.co/my-username/my-awesome-model

# Or push manually
trainer.push_to_hub(commit_message="Final model")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Model_Export]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_PyTorch]]
