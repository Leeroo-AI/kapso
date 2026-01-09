# Implementation: UnslothTrainingArguments

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|TRL SFTConfig|https://huggingface.co/docs/trl/sft_trainer]]
* [[source::Doc|Transformers TrainingArguments|https://huggingface.co/docs/transformers/main_classes/trainer]]
|-
! Domains
| [[domain::NLP]], [[domain::Training]], [[domain::Hyperparameters]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Wrapper for TRL's SFTConfig that adds Unsloth-specific training arguments for optimized fine-tuning, particularly embedding layer learning rate control.

=== Description ===

`UnslothTrainingArguments` extends TRL's `SFTConfig` (or `TrainingArguments` for older versions) with Unsloth-specific parameters. The primary addition is `embedding_learning_rate`, which enables separate learning rates for embedding layers (`embed_tokens`, `lm_head`) when training them via `modules_to_save`.

This is a Wrapper Doc - it documents how Unsloth uses and extends an external library's configuration class.

=== Usage ===

Use `UnslothTrainingArguments` when configuring training with `SFTTrainer`. It accepts all standard `SFTConfig` parameters plus Unsloth extensions. Pass the resulting config object to `SFTTrainer` via the `args` parameter.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/trainer.py
* '''Lines:''' L133-137

=== External Reference ===
* '''Library:''' [https://github.com/huggingface/trl TRL]
* '''Base Class:''' `trl.SFTConfig` or `transformers.TrainingArguments`

=== Signature ===
<syntaxhighlight lang="python">
class UnslothTrainingArguments(TrainingArguments):
    def __init__(
        self,
        # Unsloth-specific
        embedding_learning_rate: float = None,
        # Standard TrainingArguments/SFTConfig parameters
        output_dir: str = None,
        per_device_train_batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 5e-5,
        num_train_epochs: float = 3.0,
        max_steps: int = -1,
        warmup_steps: int = 0,
        warmup_ratio: float = 0.0,
        lr_scheduler_type: str = "linear",
        weight_decay: float = 0.0,
        optim: str = "adamw_torch",
        logging_steps: int = 500,
        save_steps: int = 500,
        save_total_limit: int = None,
        fp16: bool = False,
        bf16: bool = False,
        seed: int = 42,
        **kwargs,
    ):
        """
        Training arguments with Unsloth extensions.

        Args:
            embedding_learning_rate: Separate LR for embed_tokens/lm_head (if in modules_to_save)
            output_dir: Directory for checkpoints and outputs
            per_device_train_batch_size: Batch size per GPU
            gradient_accumulation_steps: Steps before gradient update
            learning_rate: Peak learning rate for LoRA parameters
            num_train_epochs: Total training epochs
            max_steps: Override epochs with fixed step count
            warmup_steps: Linear warmup steps
            lr_scheduler_type: LR schedule ("linear", "cosine", "constant")
            optim: Optimizer ("adamw_8bit" recommended for QLoRA)
            fp16/bf16: Mixed precision training

        Note: All trl.SFTConfig parameters are supported via **kwargs
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth import UnslothTrainingArguments
# Or use SFTConfig directly with embedding_learning_rate if needed
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| output_dir || str || Yes || Directory for saving checkpoints
|-
| per_device_train_batch_size || int || No (default: 8) || Batch size per GPU
|-
| gradient_accumulation_steps || int || No (default: 1) || Steps before weight update
|-
| learning_rate || float || No (default: 5e-5) || Peak learning rate
|-
| warmup_steps || int || No (default: 0) || Linear warmup steps
|-
| max_steps || int || No (default: -1) || Fixed step count (-1 for epoch-based)
|-
| optim || str || No || Optimizer name ("adamw_8bit" recommended)
|-
| embedding_learning_rate || float || No || Separate LR for embedding layers
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| config || UnslothTrainingArguments || Configuration object for SFTTrainer
|}

== Usage Examples ==

=== Standard QLoRA Training Config ===
<syntaxhighlight lang="python">
from unsloth import UnslothTrainingArguments

training_args = UnslothTrainingArguments(
    output_dir = "./outputs",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,  # Effective batch = 8
    warmup_steps = 5,
    max_steps = 60,
    learning_rate = 2e-4,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 1,
    optim = "adamw_8bit",  # Memory-efficient optimizer
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 3407,
)
</syntaxhighlight>

=== With Embedding Layer Training ===
<syntaxhighlight lang="python">
from unsloth import UnslothTrainingArguments

# When using modules_to_save = ["embed_tokens", "lm_head"]
training_args = UnslothTrainingArguments(
    output_dir = "./outputs",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    learning_rate = 2e-4,
    embedding_learning_rate = 5e-5,  # Lower LR for embeddings
    max_steps = 100,
    optim = "adamw_8bit",
    bf16 = True,
)
</syntaxhighlight>

=== Long Training Run ===
<syntaxhighlight lang="python">
from unsloth import UnslothTrainingArguments

training_args = UnslothTrainingArguments(
    output_dir = "./outputs",
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 2,
    num_train_epochs = 3,
    learning_rate = 1e-4,
    warmup_ratio = 0.03,  # 3% warmup
    lr_scheduler_type = "cosine",
    save_strategy = "epoch",
    logging_steps = 10,
    optim = "adamw_8bit",
    bf16 = True,
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_Training_Configuration]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:Unslothai_Unsloth_Sample_Packing_Tip]]
* [[uses_heuristic::Heuristic:Unslothai_Unsloth_Embedding_Learning_Rate_Tip]]
