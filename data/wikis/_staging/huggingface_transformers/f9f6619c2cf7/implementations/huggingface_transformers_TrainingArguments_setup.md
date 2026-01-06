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
Concrete configuration class for training neural networks provided by HuggingFace Transformers.

=== Description ===
The TrainingArguments class implements the training configuration principle by providing a comprehensive dataclass that captures all hyperparameters and settings needed to train transformer models. It serves as the central configuration hub passed to the Trainer class, controlling everything from learning rates and batch sizes to distributed training strategies and logging preferences.

This implementation uses Python dataclasses for type safety and default value management, making it easy to instantiate with sensible defaults while allowing fine-grained control over any parameter. It includes validation logic to ensure parameter combinations are valid and automatically handles device placement and distributed training configuration.

=== Usage ===
Import and instantiate TrainingArguments at the beginning of your training script, after defining your dataset and model but before creating the Trainer. Required parameter is output_dir for checkpoint storage. Commonly configured parameters include num_train_epochs, per_device_train_batch_size, learning_rate, and evaluation strategies. Use when fine-tuning any HuggingFace transformer model or training custom PyTorch models with the Trainer API.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/training_args.py:L198-1200

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class TrainingArguments:
    """
    TrainingArguments is the subset of the arguments we use in our example scripts **which relate to the training loop
    itself**.

    Parameters:
        output_dir (`str`, *optional*, defaults to `"trainer_output"`):
            The output directory where the model predictions and checkpoints will be written.
        do_train (`bool`, *optional*, defaults to `False`):
            Whether to run training or not.
        do_eval (`bool`, *optional*):
            Whether to run evaluation on the validation set or not.
        eval_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"no"`):
            The evaluation strategy to adopt during training. Possible values are "no", "steps", "epoch".
        per_device_train_batch_size (`int`, *optional*, defaults to 8):
            The batch size per device.
        per_device_eval_batch_size (`int`, *optional*, defaults to 8):
            The batch size per device for evaluation.
        gradient_accumulation_steps (`int`, *optional*, defaults to 1):
            Number of updates steps to accumulate the gradients for.
        learning_rate (`float`, *optional*, defaults to 5e-5):
            The initial learning rate for AdamW optimizer.
        weight_decay (`float`, *optional*, defaults to 0):
            The weight decay to apply.
        num_train_epochs(`float`, *optional*, defaults to 3.0):
            Total number of training epochs to perform.
        max_steps (`int`, *optional*, defaults to -1):
            If set to a positive number, the total number of training steps to perform.
        lr_scheduler_type (`str` or [`SchedulerType`], *optional*, defaults to `"linear"`):
            The scheduler type to use.
        warmup_steps (`int` or `float`, *optional*, defaults to 0):
            Number of steps used for a linear warmup.
        logging_dir (`str`, *optional*):
            TensorBoard log directory.
        logging_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"steps"`):
            The logging strategy to adopt during training.
        logging_steps (`int`, *optional*, defaults to 500):
            Number of update steps between two logs.
        save_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"steps"`):
            The checkpoint save strategy to adopt during training.
        save_steps (`int`, *optional*, defaults to 500):
            Number of updates steps before two checkpoint saves.
        eval_steps (`int`, *optional*):
            Number of update steps between two evaluations.
        fp16 (`bool`, *optional*, defaults to `False`):
            Whether to use fp16 16-bit (mixed) precision training.
        bf16 (`bool`, *optional*, defaults to `False`):
            Whether to use bf16 16-bit (mixed) precision training.
        **kwargs:
            Additional keyword arguments for advanced configuration.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import TrainingArguments
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| output_dir || str || Yes || Directory path where model checkpoints and predictions will be saved
|-
| num_train_epochs || float || No || Number of complete passes through training dataset (default: 3.0)
|-
| per_device_train_batch_size || int || No || Batch size per GPU/TPU/CPU for training (default: 8)
|-
| per_device_eval_batch_size || int || No || Batch size per device for evaluation (default: 8)
|-
| gradient_accumulation_steps || int || No || Accumulate gradients over N steps before updating (default: 1)
|-
| learning_rate || float || No || Initial learning rate for optimizer (default: 5e-5)
|-
| weight_decay || float || No || Weight decay coefficient for regularization (default: 0)
|-
| max_steps || int || No || Override num_train_epochs if positive (default: -1)
|-
| lr_scheduler_type || str || No || Type of learning rate scheduler: "linear", "cosine", etc. (default: "linear")
|-
| warmup_ratio || float || No || Ratio of total training steps for warmup (default: 0.0)
|-
| warmup_steps || int || No || Absolute number of warmup steps (default: 0)
|-
| logging_strategy || str || No || When to log: "no", "steps", "epoch" (default: "steps")
|-
| logging_steps || int || No || Log every N update steps (default: 500)
|-
| save_strategy || str || No || When to save checkpoints: "no", "steps", "epoch" (default: "steps")
|-
| save_steps || int || No || Save checkpoint every N update steps (default: 500)
|-
| eval_strategy || str || No || When to evaluate: "no", "steps", "epoch" (default: "no")
|-
| eval_steps || int || No || Evaluate every N update steps (default: None)
|-
| fp16 || bool || No || Use 16-bit floating point precision (default: False)
|-
| bf16 || bool || No || Use bfloat16 precision (default: False)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| training_args || TrainingArguments || Configured training arguments object ready to pass to Trainer
|}

== Usage Examples ==

=== Basic Training Configuration ===
<syntaxhighlight lang="python">
from transformers import TrainingArguments

# Minimal configuration with required parameter
training_args = TrainingArguments(
    output_dir="./results"
)

# Standard configuration for fine-tuning
training_args = TrainingArguments(
    output_dir="./fine-tuned-model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)
</syntaxhighlight>

=== Advanced Configuration with Mixed Precision ===
<syntaxhighlight lang="python">
from transformers import TrainingArguments

# Production training with gradient accumulation and mixed precision
training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,  # Effective batch size: 8 * 4 = 32
    learning_rate=5e-5,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    fp16=True,  # Enable mixed precision training
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,  # Keep only 3 best checkpoints
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="tensorboard"
)
</syntaxhighlight>

=== Configuration for Large Models ===
<syntaxhighlight lang="python">
from transformers import TrainingArguments

# Configuration optimized for large models with memory constraints
training_args = TrainingArguments(
    output_dir="./large-model-checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # Effective batch size: 32
    learning_rate=1e-4,
    weight_decay=0.1,
    lr_scheduler_type="cosine",
    warmup_steps=1000,
    bf16=True,  # Use bfloat16 for better numerical stability
    gradient_checkpointing=True,  # Save memory at cost of speed
    max_grad_norm=1.0,  # Gradient clipping
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    logging_steps=10,
    dataloader_num_workers=4,
    remove_unused_columns=False,
    optim="adamw_torch",  # Specify optimizer
    group_by_length=True,  # Efficiency for variable length sequences
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_TrainingArguments_Configuration]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Training_Environment]]
