{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|Transformers Trainer|https://huggingface.co/docs/transformers/main_classes/trainer]]
|-
! Domains
| [[domain::Training]], [[domain::NLP]], [[domain::Fine_Tuning]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Concrete tool for executing the training loop using HuggingFace Trainer with PEFT models for LoRA fine-tuning.

=== Description ===

`Trainer.train()` executes the full training loop for the PEFT model. HuggingFace's Trainer handles all training infrastructure: batching, gradient accumulation, mixed precision, logging, checkpointing, and evaluation. PEFT models work seamlessly with Trainer since only adapter parameters have `requires_grad=True`.

=== Usage ===

Use this after setting up your PEFT model, tokenized dataset, and TrainingArguments. The Trainer handles the training loop automatically. For LoRA fine-tuning, typical settings include lower learning rates (1e-4 to 2e-4), gradient accumulation for larger effective batch sizes, and regular evaluation.

== Code Reference ==

=== Source Location ===
* '''Library:''' [https://github.com/huggingface/transformers transformers]
* '''Class:''' `transformers.Trainer`

=== Signature ===
<syntaxhighlight lang="python">
def train(
    self,
    resume_from_checkpoint: Optional[Union[str, bool]] = None,
    trial: Optional["optuna.Trial"] = None,
    ignore_keys_for_eval: Optional[List[str]] = None,
    **kwargs,
) -> TrainOutput:
    """
    Main training entry point.

    Args:
        resume_from_checkpoint: Path to checkpoint or bool to auto-detect
        trial: Optuna trial for hyperparameter search
        ignore_keys_for_eval: Keys to ignore during evaluation

    Returns:
        TrainOutput: Contains global_step, training_loss, metrics
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || PeftModel || Yes || PEFT-wrapped model (passed to Trainer constructor)
|-
| args || TrainingArguments || Yes || Training hyperparameters and configuration
|-
| train_dataset || Dataset || Yes || Tokenized training data
|-
| eval_dataset || Dataset || No || Tokenized evaluation data
|-
| data_collator || DataCollator || No || Batch collation function
|-
| resume_from_checkpoint || str or bool || No || Checkpoint to resume from
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| TrainOutput || dataclass || Contains global_step, training_loss, and metrics dict
|-
| checkpoints || Files || Saved to output_dir based on save_strategy
|-
| logs || Files || Training logs to logging_dir
|}

== Usage Examples ==

=== Standard LoRA Training ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig

# 1. Create PEFT model
model = get_peft_model(base_model, lora_config)

# 2. Define training arguments
training_args = TrainingArguments(
    output_dir="./lora-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 16
    learning_rate=2e-4,             # Typical for LoRA
    warmup_ratio=0.03,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=True,                      # Mixed precision
)

# 3. Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
)

# 4. Train
trainer.train()

# 5. Save final adapter
model.save_pretrained("./final-adapter")
</syntaxhighlight>

=== With Custom Callbacks ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments, TrainerCallback

class SaveBestCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if metrics.get("eval_loss", float("inf")) < state.best_metric:
            control.should_save = True
        return control

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[SaveBestCallback()],
)
</syntaxhighlight>

=== Resume from Checkpoint ===
<syntaxhighlight lang="python">
# Resume training from checkpoint
trainer.train(resume_from_checkpoint="./lora-output/checkpoint-500")

# Or auto-detect latest checkpoint
trainer.train(resume_from_checkpoint=True)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_peft_Training_Execution]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
