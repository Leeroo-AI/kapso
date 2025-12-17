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

Concrete dataclass for configuring the training loop provided by the HuggingFace Transformers library.

=== Description ===

TrainingArguments is a comprehensive dataclass that encapsulates all configuration parameters needed for training transformer models. It handles hyperparameters, distributed training settings, logging, evaluation, checkpointing, and hardware optimization options. This class serves as the central configuration hub that controls every aspect of the training process in the Trainer API.

=== Usage ===

Use TrainingArguments when you need to configure training parameters for fine-tuning or training transformer models. It is the required first step before initializing a Trainer instance, allowing you to specify output directories, learning rates, batch sizes, training epochs, evaluation strategies, and advanced features like gradient accumulation, mixed precision training, and distributed training configurations.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/training_args.py
* '''Lines:''' 198-2809

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class TrainingArguments:
    output_dir: str = "trainer_output"
    do_train: bool = False
    do_eval: bool | None = None
    do_predict: bool = False
    eval_strategy: str | IntervalStrategy = "no"
    prediction_loss_only: bool = False
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    eval_accumulation_steps: int | None = None
    eval_delay: float | None = None
    torch_empty_cache_steps: int | None = None
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    num_train_epochs: float = 3.0
    max_steps: int = -1
    lr_scheduler_type: str | SchedulerType = "linear"
    lr_scheduler_kwargs: dict | str | None = None
    warmup_steps: int | float = 0
    # ... many more parameters
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
| output_dir || str || No (default: "trainer_output") || The output directory where model predictions and checkpoints will be written
|-
| learning_rate || float || No (default: 5e-5) || The initial learning rate for AdamW optimizer
|-
| per_device_train_batch_size || int || No (default: 8) || The batch size per device during training
|-
| per_device_eval_batch_size || int || No (default: 8) || The batch size per device during evaluation
|-
| num_train_epochs || float || No (default: 3.0) || Total number of training epochs to perform
|-
| max_steps || int || No (default: -1) || If set to positive number, total training steps (overrides num_train_epochs)
|-
| weight_decay || float || No (default: 0.0) || The weight decay to apply to all layers except bias and LayerNorm
|-
| eval_strategy || str or IntervalStrategy || No (default: "no") || The evaluation strategy: "no", "steps", or "epoch"
|-
| gradient_accumulation_steps || int || No (default: 1) || Number of updates steps to accumulate gradients before backward pass
|-
| warmup_steps || int or float || No (default: 0) || Number of steps for linear warmup from 0 to learning_rate
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| training_args || TrainingArguments || Configured TrainingArguments dataclass instance ready to be passed to Trainer
|}

== Usage Examples ==

=== Basic Usage ===
<syntaxhighlight lang="python">
from transformers import TrainingArguments

# Minimal configuration
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    learning_rate=5e-5
)

# Advanced configuration with evaluation and checkpointing
training_args = TrainingArguments(
    output_dir="./my_model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    warmup_steps=500,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,  # Enable mixed precision training
    gradient_accumulation_steps=4
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Training_Arguments]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_PyTorch]]
