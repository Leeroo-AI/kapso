{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|HuggingFace TrainingArguments|https://huggingface.co/docs/transformers/main_classes/trainer]]
* [[source::Repo|Transformers GitHub|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Configuration]], [[domain::Training]], [[domain::Deep_Learning]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Base configuration class for Trainer hyperparameters provided by HuggingFace Transformers library.

=== Description ===
`TrainingArguments` is the foundational configuration class for all HuggingFace Trainer-based training. It defines hyperparameters for learning rate, batch size, optimization, logging, checkpointing, and hardware utilization. While `SFTConfig` is preferred for fine-tuning, understanding `TrainingArguments` is essential as it forms the base.

=== Usage ===
Import this class when using the base HuggingFace `Trainer` directly or when you need fine-grained control over training arguments. In Unsloth workflows, prefer `SFTConfig` which extends this class with SFT-specific options.

== Code Signature ==
<syntaxhighlight lang="python">
from transformers import TrainingArguments

class TrainingArguments:
    def __init__(
        self,
        output_dir: str,
        # Training hyperparameters
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 5e-5,
        num_train_epochs: float = 3.0,
        max_steps: int = -1,
        # Optimizer
        optim: str = "adamw_torch",
        warmup_steps: int = 0,
        warmup_ratio: float = 0.0,
        lr_scheduler_type: str = "linear",
        weight_decay: float = 0.0,
        # Precision
        fp16: bool = False,
        bf16: bool = False,
        # Logging & Saving
        logging_steps: int = 500,
        save_steps: int = 500,
        save_total_limit: Optional[int] = None,
        # Evaluation
        evaluation_strategy: str = "no",
        eval_steps: Optional[int] = None,
        # Hardware
        dataloader_num_workers: int = 0,
        # Reproducibility
        seed: int = 42,
        **kwargs
    ):
        ...
</syntaxhighlight>

== I/O Contract ==
* **Consumes:**
    * Hyperparameter values for training configuration
* **Produces:**
    * Configured `TrainingArguments` instance

== Key Parameters ==
{| class="wikitable"
! Parameter !! Type !! Description
|-
|| output_dir || str || Directory for checkpoints and logs
|-
|| learning_rate || float || Initial learning rate (2e-4 for LoRA)
|-
|| per_device_train_batch_size || int || Batch size per GPU
|-
|| gradient_accumulation_steps || int || Steps before weight update
|-
|| optim || str || Optimizer ("adamw_8bit" recommended)
|-
|| max_steps || int || Total training steps (-1 for epoch-based)
|-
|| warmup_steps || int || LR warmup steps
|}

== Related Pages ==
=== Context & Requirements ===
* [[requires_env::Environment:Unsloth_CUDA_Environment]]

=== Tips and Tricks ===
* [[uses_heuristic::Heuristic:Learning_Rate_Tuning]]
* [[uses_heuristic::Heuristic:Batch_Size_Optimization]]
* [[uses_heuristic::Heuristic:Warmup_Steps_Heuristic]]
* [[uses_heuristic::Heuristic:Gradient_Accumulation_Strategy]]

