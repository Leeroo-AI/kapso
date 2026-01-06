{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Trainer Documentation|https://huggingface.co/docs/transformers/main_classes/trainer]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Training]], [[domain::Model_Persistence]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Concrete checkpoint and model persistence methods provided by HuggingFace Transformers Trainer.

=== Description ===
The save_model() method implements the checkpoint saving principle by providing a comprehensive model persistence mechanism that handles the complexities of saving transformer models across various training configurations. It intelligently manages model state serialization, handles distributed training scenarios (FSDP, DeepSpeed, model parallelism), respects device placement constraints, and saves models in the standard HuggingFace format for easy loading and deployment.

This implementation automatically detects the training environment and applies the appropriate saving strategy, whether single GPU, multi-GPU with DDP, FSDP sharding, DeepSpeed ZeRO, or TPU training. It saves model weights, configuration files, and optionally the tokenizer, creating a complete package for model reuse. The Trainer also provides save_state() functionality (imported from trainer_pt_utils) for saving full training state including optimizer and scheduler states.

=== Usage ===
The save_model() method is typically called automatically by Trainer based on save_strategy in TrainingArguments, but can also be invoked manually. Call it to save the model at any point during or after training, pass output_dir to specify location (defaults to args.output_dir), or call without arguments to save to the default location. Use save_model() for deployment-ready model saving (just weights + config) and rely on automatic checkpoint saving for full training state preservation during training.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/trainer.py:L3989-4052 (save_model), trainer_pt_utils.py (save_state)

=== Signature ===
<syntaxhighlight lang="python">
def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
    """
    Will save the model, so you can reload it using `from_pretrained()`.

    Will only save from the main process.

    Args:
        output_dir (str, optional): Directory to save the model. Defaults to self.args.output_dir.
        _internal_call (bool): Internal flag used by Trainer to distinguish between
                               automatic saves and user-initiated saves.
    """

def save_state(self):
    """
    Saves the Trainer state (epoch, step, optimizer, scheduler, RNG states) to the output directory.
    This method is imported from trainer_pt_utils and called automatically during checkpointing.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import Trainer
# save_model() and save_state() are methods of Trainer class
</syntaxhighlight>

== I/O Contract ==

=== Inputs (save_model) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| output_dir || str || No || Directory path where model will be saved (defaults to self.args.output_dir)
|-
| _internal_call || bool || No || Internal flag to distinguish automatic from manual saves (default: False)
|}

=== Inputs (save_state) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| (none) || - || - || Uses self.state and self.args.output_dir from trainer instance
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| (files on disk) || - || Model weights, config, and optionally tokenizer saved to output_dir
|}

== Usage Examples ==

=== Automatic Checkpoint Saving During Training ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Configure automatic saving
training_args = TrainingArguments(
    output_dir="./training-checkpoints",
    num_train_epochs=5,
    save_strategy="steps",     # Save every N steps
    save_steps=1000,           # Save every 1000 steps
    save_total_limit=3,        # Keep only 3 most recent checkpoints
    logging_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Trainer automatically calls save_model() every 1000 steps
# Creates: checkpoint-1000, checkpoint-2000, checkpoint-3000, etc.
trainer.train()

# Final model automatically saved to output_dir
</syntaxhighlight>

=== Manual Model Saving ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
training_args = TrainingArguments(output_dir="./training")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Train for some steps
trainer.train()

# Manually save the model to a specific location
trainer.save_model("./my-fine-tuned-model")

# Saved files:
# - ./my-fine-tuned-model/pytorch_model.bin (or model.safetensors)
# - ./my-fine-tuned-model/config.json
# - ./my-fine-tuned-model/training_args.bin

# Can now load with from_pretrained
from transformers import AutoModelForCausalLM
loaded_model = AutoModelForCausalLM.from_pretrained("./my-fine-tuned-model")
</syntaxhighlight>

=== Saving with Tokenizer ===
<syntaxhighlight lang="python">
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)

model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
tokenizer = AutoTokenizer.from_pretrained("t5-small")

training_args = TrainingArguments(
    output_dir="./t5-fine-tuned",
    num_train_epochs=3,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer  # Tokenizer will be saved with model
)

trainer.train()

# Final model includes tokenizer
# ./t5-fine-tuned/
#   - pytorch_model.bin
#   - config.json
#   - tokenizer_config.json
#   - spiece.model (tokenizer vocabulary)
#   - special_tokens_map.json

# Can load both together
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("./t5-fine-tuned")
tokenizer = AutoTokenizer.from_pretrained("./t5-fine-tuned")
</syntaxhighlight>

=== Best Model Saving ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=3)

training_args = TrainingArguments(
    output_dir="./best-model-training",
    num_train_epochs=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,       # Load best checkpoint when done
    metric_for_best_model="eval_f1",   # Track F1 score
    greater_is_better=True,
    save_total_limit=2                 # Keep best + latest
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Trains and tracks best model based on eval_f1
trainer.train()

# At end, model is the best checkpoint (not necessarily the last epoch)
# Save this best model for deployment
trainer.save_model("./best-model-final")
</syntaxhighlight>

=== Checkpoint Directory Structure ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments
import os

model = AutoModelForCausalLM.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir="./checkpointing-demo",
    num_train_epochs=3,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=None  # Keep all checkpoints
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
trainer.train()

# Resulting directory structure:
# ./checkpointing-demo/
#   checkpoint-500/
#     - pytorch_model.bin
#     - config.json
#     - optimizer.pt
#     - scheduler.pt
#     - trainer_state.json
#     - training_args.bin
#     - rng_state.pth
#   checkpoint-1000/
#     - (same structure)
#   checkpoint-1500/
#     - (same structure)
#   (final model files)

# List all checkpoints
checkpoints = [d for d in os.listdir("./checkpointing-demo")
               if d.startswith("checkpoint-")]
print(f"Saved checkpoints: {sorted(checkpoints)}")
</syntaxhighlight>

=== Resuming from Checkpoint ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./resume-training",
    num_train_epochs=10,
    save_strategy="steps",
    save_steps=1000
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Initial training (may be interrupted)
trainer.train()

# To resume from latest checkpoint
trainer_resumed = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Option 1: Auto-detect last checkpoint
trainer_resumed.train(resume_from_checkpoint=True)

# Option 2: Specify checkpoint
trainer_resumed.train(resume_from_checkpoint="./resume-training/checkpoint-3000")

# Training continues from where it left off
# - Model weights restored
# - Optimizer state restored (momentum buffers, etc.)
# - Scheduler state restored
# - Global step counter continues
# - RNG state restored for reproducibility
</syntaxhighlight>

=== Saving for HuggingFace Hub ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./hub-model",
    num_train_epochs=3,
    push_to_hub=True,              # Automatically push to Hub
    hub_model_id="my-username/my-fine-tuned-bert",
    hub_strategy="every_save",     # Push on every save
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Train and automatically push to HuggingFace Hub
trainer.train()

# Model available at: https://huggingface.co/my-username/my-fine-tuned-bert

# Or push manually after training
trainer.push_to_hub(commit_message="Fine-tuned BERT on my dataset")
</syntaxhighlight>

=== Custom Save Behavior with Callbacks ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments, TrainerCallback
import shutil

class CustomSaveCallback(TrainerCallback):
    """Save model in multiple formats or locations."""

    def on_save(self, args, state, control, **kwargs):
        """Called whenever trainer saves a checkpoint."""
        checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"

        # Create additional backup
        backup_dir = f"/backup/checkpoint-{state.global_step}"
        if os.path.exists(checkpoint_dir):
            shutil.copytree(checkpoint_dir, backup_dir, dirs_exist_ok=True)
            print(f"Backed up checkpoint to {backup_dir}")

        return control

training_args = TrainingArguments(
    output_dir="./custom-save",
    save_strategy="steps",
    save_steps=1000
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[CustomSaveCallback()]
)

trainer.train()
</syntaxhighlight>

=== Saving Large Models with Sharding ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM

# Large model (e.g., 13B parameters)
model = AutoModelForCausalLM.from_pretrained("large-model-13b")

training_args = TrainingArguments(
    output_dir="./large-model-checkpoints",
    num_train_epochs=1,
    save_strategy="steps",
    save_steps=1000,
    # Trainer automatically shards large models
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()

# Saved as sharded checkpoint:
# ./checkpoint-1000/
#   - pytorch_model-00001-of-00004.bin (shard 1)
#   - pytorch_model-00002-of-00004.bin (shard 2)
#   - pytorch_model-00003-of-00004.bin (shard 3)
#   - pytorch_model-00004-of-00004.bin (shard 4)
#   - pytorch_model.bin.index.json (mapping)

# Loading automatically handles sharding
loaded_model = AutoModelForCausalLM.from_pretrained("./checkpoint-1000")
</syntaxhighlight>

=== SafeTensors Format ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./safetensors-model",
    num_train_epochs=3,
    save_strategy="epoch",
    save_safetensors=True  # Use SafeTensors format (faster, safer)
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()

# Saved as model.safetensors instead of pytorch_model.bin
# Benefits:
# - Faster loading
# - More secure (no pickle)
# - Better for production deployment

# Loading works the same
loaded_model = AutoModelForSequenceClassification.from_pretrained("./safetensors-model")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Checkpoint_Saving]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Training_Environment]]
