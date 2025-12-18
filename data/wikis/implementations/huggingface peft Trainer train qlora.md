{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Trainer|https://huggingface.co/docs/transformers/main_classes/trainer]]
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
|-
! Domains
| [[domain::Fine_Tuning]], [[domain::Quantization]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Concrete tool for training QLoRA models using the HuggingFace Trainer with quantization-appropriate settings.

=== Description ===

This uses the same `Trainer.train()` method as standard LoRA but with `TrainingArguments` configured for quantized model training. Key settings include gradient accumulation, paged optimizers, and mixed precision training.

=== Usage ===

Configure TrainingArguments for QLoRA, then call `trainer.train()` as usual.

== Code Reference ==

=== Source Location ===
* '''Library:''' `transformers.Trainer` (external)
* '''Method:''' `train()`

=== Signature ===
<syntaxhighlight lang="python">
def train(
    self,
    resume_from_checkpoint: Optional[Union[str, bool]] = None,
    trial: Optional[Any] = None,
    ignore_keys_for_eval: Optional[list[str]] = None,
    **kwargs,
) -> TrainOutput:
    """Execute training loop."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments
</syntaxhighlight>

== Usage Examples ==

=== QLoRA Training Configuration ===
<syntaxhighlight lang="python">
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./qlora-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch = 16
    learning_rate=2e-4,
    fp16=True,  # Or bf16=True for bfloat16
    optim="paged_adamw_8bit",  # Memory-efficient optimizer
    logging_steps=10,
    save_strategy="epoch",
    gradient_checkpointing=True,  # Additional memory savings
)

trainer = Trainer(
    model=qlora_model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

trainer.train()
</syntaxhighlight>

=== Monitoring for Stability ===
<syntaxhighlight lang="python">
# If training shows NaN losses, try:
training_args = TrainingArguments(
    # ... other args ...
    max_grad_norm=0.3,  # Gradient clipping
    warmup_ratio=0.03,  # Learning rate warmup
    lr_scheduler_type="cosine",
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_peft_QLoRA_Training_Execution]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_peft_Quantization_Environment]]
