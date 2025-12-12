{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|TRL GitHub|https://github.com/huggingface/trl]]
* [[source::Doc|TRL SFTConfig|https://huggingface.co/docs/trl/sft_trainer]]
* [[source::Doc|HuggingFace TrainingArguments|https://huggingface.co/docs/transformers/main_classes/trainer]]
|-
! Domains
| [[domain::Configuration]], [[domain::Training]], [[domain::LLMs]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Configuration class for SFTTrainer hyperparameters provided by HuggingFace TRL library.

=== Description ===
`SFTConfig` extends `TrainingArguments` with SFT-specific options like `packing` and `max_seq_length`. It provides a clean interface for setting all training hyperparameters including learning rate, batch size, optimization settings, and logging configuration. Fully compatible with Unsloth workflows.

=== Usage ===
Import this class to configure training hyperparameters for `SFTTrainer`. Preferred over raw `TrainingArguments` when using TRL for fine-tuning. Set parameters based on heuristics for your specific task and hardware.

== Code Signature ==
<syntaxhighlight lang="python">
from trl import SFTConfig

class SFTConfig(TrainingArguments):
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
        # Optimizer settings
        optim: str = "adamw_torch",
        warmup_steps: int = 0,
        warmup_ratio: float = 0.0,
        lr_scheduler_type: str = "linear",
        weight_decay: float = 0.0,
        # SFT-specific
        max_seq_length: Optional[int] = 1024,
        packing: bool = False,
        dataset_text_field: Optional[str] = None,
        # Logging
        logging_steps: int = 500,
        save_steps: int = 500,
        eval_steps: int = 500,
        # Others
        fp16: bool = False,
        bf16: bool = False,
        seed: int = 42,
        **kwargs
    ):
        ...
</syntaxhighlight>

== I/O Contract ==
* **Consumes:**
    * Hyperparameter values (learning rate, batch size, etc.)
    * Path strings (output_dir, logging_dir)
* **Produces:**
    * Configured `SFTConfig` instance for use with `SFTTrainer`

== Example Usage ==
<syntaxhighlight lang="python">
from trl import SFTConfig

# Recommended configuration for Unsloth QLoRA
args = SFTConfig(
    output_dir = "outputs",
    
    # Batch settings
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    
    # Learning rate
    learning_rate = 2e-4,
    lr_scheduler_type = "linear",
    warmup_steps = 10,
    
    # Training duration
    max_steps = 60,  # or num_train_epochs = 1
    
    # Optimizer
    optim = "adamw_8bit",
    weight_decay = 0.01,
    
    # SFT-specific
    max_seq_length = 2048,
    packing = False,
    
    # Logging
    logging_steps = 1,
    save_steps = 25,
    
    # Reproducibility
    seed = 3407,
)
</syntaxhighlight>

== Related Pages ==
=== Context & Requirements ===
* [[requires_env::Environment:Unsloth_CUDA_Environment]]

=== Tips and Tricks ===
* [[uses_heuristic::Heuristic:Learning_Rate_Tuning]]
* [[uses_heuristic::Heuristic:Batch_Size_Optimization]]
* [[uses_heuristic::Heuristic:Warmup_Steps_Heuristic]]
* [[uses_heuristic::Heuristic:AdamW_8bit_Optimizer_Usage]]
* [[uses_heuristic::Heuristic:Gradient_Accumulation_Strategy]]

