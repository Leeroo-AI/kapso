# Implementation: SFTConfig

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|TRL SFTTrainer|https://huggingface.co/docs/trl/sft_trainer]]
* [[source::Doc|HuggingFace TrainingArguments|https://huggingface.co/docs/transformers/main_classes/trainer]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Training]], [[domain::Configuration]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

Concrete tool for configuring supervised fine-tuning parameters provided by TRL and extended by Unsloth.

=== Description ===

`SFTConfig` (from TRL) and `UnslothTrainingArguments` define the hyperparameters for training. These configuration classes control:

* Learning rate and scheduling
* Batch sizes and gradient accumulation
* Training duration (epochs/steps)
* Checkpointing and logging
* Optimizer settings
* Sample packing and padding-free training

Unsloth extends the base configuration with `embedding_learning_rate` for separate embedding optimization and automatic padding-free training detection.

=== Usage ===

Import this configuration class when setting up training hyperparameters. Create an instance with your desired settings and pass it to SFTTrainer. This is the standard way to configure training in the HuggingFace/TRL ecosystem.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/trainer.py
* '''Lines:''' 133-198 (UnslothTrainingArguments), inherits from TRL SFTConfig

=== Signature ===
<syntaxhighlight lang="python">
# From TRL (wrapped by Unsloth)
class SFTConfig(TrainingArguments):
    def __init__(
        self,
        output_dir: str,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 5e-5,
        num_train_epochs: float = 3.0,
        max_steps: int = -1,
        warmup_ratio: float = 0.0,
        warmup_steps: int = 0,
        logging_steps: int = 500,
        save_steps: int = 500,
        save_total_limit: Optional[int] = None,
        seed: int = 42,
        bf16: bool = False,
        fp16: bool = False,
        optim: str = "adamw_torch",
        weight_decay: float = 0.0,
        lr_scheduler_type: str = "linear",
        max_seq_length: int = 1024,
        packing: bool = False,
        dataset_text_field: str = "text",
        **kwargs,
    ):
        """
        Configuration for supervised fine-tuning.

        Args:
            output_dir: Directory for checkpoints and logs
            per_device_train_batch_size: Batch size per GPU
            gradient_accumulation_steps: Steps to accumulate gradients
            learning_rate: Initial learning rate
            num_train_epochs: Total training epochs
            max_steps: Maximum steps (overrides epochs if set)
            warmup_ratio: Fraction of steps for warmup
            logging_steps: Log every N steps
            save_steps: Save checkpoint every N steps
            bf16/fp16: Mixed precision training
            optim: Optimizer type
            weight_decay: L2 regularization
            max_seq_length: Maximum sequence length
            packing: Enable sequence packing
        """

# Unsloth extension
class UnslothTrainingArguments(SFTConfig):
    def __init__(
        self,
        embedding_learning_rate: float = None,
        *args,
        **kwargs,
    ):
        """
        Extended training arguments with Unsloth-specific features.

        Args:
            embedding_learning_rate: Separate LR for embed_tokens/lm_head
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from trl import SFTConfig

# Or with Unsloth extensions
from unsloth import UnslothTrainingArguments
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| output_dir || str || Yes || Directory for saving checkpoints and logs
|-
| per_device_train_batch_size || int || No || Batch size per GPU (default: 8)
|-
| gradient_accumulation_steps || int || No || Steps to accumulate gradients (default: 1)
|-
| learning_rate || float || No || Initial learning rate (default: 5e-5)
|-
| num_train_epochs || float || No || Number of training epochs (default: 3.0)
|-
| max_steps || int || No || Maximum training steps, -1 for unlimited (default: -1)
|-
| warmup_ratio || float || No || Fraction of steps for LR warmup (default: 0.0)
|-
| logging_steps || int || No || Log metrics every N steps (default: 500)
|-
| save_steps || int || No || Save checkpoint every N steps (default: 500)
|-
| bf16/fp16 || bool || No || Enable mixed precision (default: False)
|-
| packing || bool || No || Enable sequence packing (default: False)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| config || SFTConfig || Configuration object for SFTTrainer
|}

== Usage Examples ==

=== Basic Configuration ===
<syntaxhighlight lang="python">
from trl import SFTConfig

# Standard training configuration
training_args = SFTConfig(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    warmup_ratio=0.03,
    logging_steps=10,
    save_steps=100,
    bf16=True,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
)
</syntaxhighlight>

=== With Sequence Packing ===
<syntaxhighlight lang="python">
from trl import SFTConfig

# Enable sequence packing for efficiency
training_args = SFTConfig(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    max_seq_length=2048,
    packing=True,  # Pack multiple samples per sequence
    bf16=True,
    logging_steps=10,
)
</syntaxhighlight>

=== With Unsloth Embedding LR ===
<syntaxhighlight lang="python">
from unsloth import UnslothTrainingArguments

# Separate learning rate for embeddings
training_args = UnslothTrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    embedding_learning_rate=5e-5,  # Lower LR for embeddings
    num_train_epochs=3,
    warmup_ratio=0.03,
    bf16=True,
)
</syntaxhighlight>

=== Production Configuration ===
<syntaxhighlight lang="python">
from trl import SFTConfig

# Production-ready settings
training_args = SFTConfig(
    output_dir="./checkpoints",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    warmup_ratio=0.03,
    warmup_steps=0,
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,  # Keep only last 3 checkpoints
    bf16=True,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    max_seq_length=2048,
    packing=True,
    seed=3407,
    report_to="wandb",  # Enable W&B logging
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_Training_Configuration]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_TRL]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:Unslothai_Unsloth_Batch_Size_Selection]]
