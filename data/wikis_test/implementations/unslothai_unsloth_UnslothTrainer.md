# Implementation: UnslothTrainer

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|TRL Documentation|https://huggingface.co/docs/trl]]
|-
! Domains
| [[domain::Training]], [[domain::Fine_Tuning]], [[domain::NLP]]
|-
! Last Updated
| [[last_updated::2025-12-15 19:00 GMT]]
|}

== Overview ==
Concrete tool for training language models with embedding learning rate support and optimized training loops provided by the Unsloth library.

=== Description ===
`UnslothTrainer` is a custom extension of TRL's `SFTTrainer` that provides additional functionality for Unsloth's optimized training:

1. **Embedding Learning Rate** - Separate learning rate for embedding layers when training `embed_tokens` and `lm_head`
2. **Backwards Compatibility** - Automatic patches for TRL version compatibility
3. **Auto Sample Packing** - Automatic sample packing configuration for efficient batching
4. **Auto Padding-Free** - Automatic padding-free batching when supported

The trainer integrates with Unsloth's kernel optimizations and handles the complexity of training embeddings at a different learning rate than LoRA adapters.

=== Usage ===
Use this class when you need to:
- Train models with different learning rates for embeddings vs LoRA
- Use `UnslothTrainingArguments` for embedding-specific configuration
- Leverage Unsloth's training optimizations

For most use cases, the standard `SFTTrainer` with Unsloth's patches works fine. Use `UnslothTrainer` specifically for embedding learning rate control.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unslothai/unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/trainer.py#L181-L198 unsloth/trainer.py]
* '''Lines:''' 181-198

Source Files: unsloth/trainer.py:L181-L198; unsloth/trainer.py:L132-L136

=== Signature ===
<syntaxhighlight lang="python">
class UnslothTrainingArguments(TrainingArguments):
    """Extended training arguments with embedding learning rate support."""
    def __init__(
        self,
        embedding_learning_rate: float = None,
        *args,
        **kwargs,
    ):
        """
        Args:
            embedding_learning_rate: Separate learning rate for embed_tokens/lm_head
            *args: Passed to TrainingArguments
            **kwargs: Passed to TrainingArguments
        """


class UnslothTrainer(SFTTrainer):
    """SFTTrainer extension with embedding learning rate support."""

    def create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer with separate parameter groups for embeddings.

        If embedding_learning_rate is set in args, creates optimizer
        with two parameter groups:
        1. Non-embedding parameters: use standard learning rate
        2. Embedding parameters (modules_to_save): use embedding_learning_rate

        Returns:
            Optimizer configured for the training setup
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth.trainer import UnslothTrainer, UnslothTrainingArguments
</syntaxhighlight>

== I/O Contract ==

=== Inputs (UnslothTrainingArguments) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| embedding_learning_rate || float || No || Learning rate for embed_tokens/lm_head
|-
| output_dir || str || Yes || Directory for checkpoints
|-
| per_device_train_batch_size || int || No || Batch size per GPU
|-
| learning_rate || float || No || Learning rate for LoRA parameters
|-
| num_train_epochs || int || No || Number of training epochs
|-
| (all other TrainingArguments) || various || No || Standard HuggingFace args
|}

=== Inputs (UnslothTrainer) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || PreTrainedModel || Yes || Model with LoRA adapters
|-
| tokenizer || PreTrainedTokenizer || Yes || Tokenizer for data processing
|-
| train_dataset || Dataset || Yes || Training data
|-
| args || UnslothTrainingArguments || Yes || Training configuration
|-
| data_collator || DataCollator || No || Data collation function
|-
| dataset_text_field || str || No || Field containing text data
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| trainer.train() || TrainOutput || Training statistics (loss, steps, etc.)
|-
| checkpoints || Files || Model checkpoints in output_dir
|-
| logs || Dict || Training metrics and logs
|}

== Usage Examples ==

=== Basic Usage with Embedding Learning Rate ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
from unsloth.trainer import UnslothTrainer, UnslothTrainingArguments
from datasets import load_dataset

# Load model with embedding training
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Llama-3.2-1B-Instruct",
    load_in_4bit=True,
)

# Include embeddings in training
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                    "embed_tokens", "lm_head"],  # Train embeddings
)

# Configure with separate embedding learning rate
args = UnslothTrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,              # For LoRA parameters
    embedding_learning_rate=5e-5,    # Lower rate for embeddings
    num_train_epochs=1,
    fp16=True,
    logging_steps=10,
)

# Create trainer
trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=args,
)

# Train
trainer.train()
</syntaxhighlight>

=== Standard SFTTrainer Usage (Recommended for Most Cases) ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# Load and configure model
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Llama-3.2-1B-Instruct",
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(model, r=16)

# Standard SFTTrainer works with Unsloth patches
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=SFTConfig(
        output_dir="outputs",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        fp16=True,
    ),
)

trainer.train()
</syntaxhighlight>

=== With Sample Packing ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Llama-3.2-1B-Instruct",
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=16)

# Enable sample packing in SFTConfig
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="outputs",
        packing=True,  # Enable sample packing
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        max_steps=100,
    ),
)

# Unsloth auto-enables padding-free if packing is set
trainer.train()
</syntaxhighlight>

=== Custom Optimizer Setup ===
<syntaxhighlight lang="python">
from unsloth.trainer import UnslothTrainer, UnslothTrainingArguments

# The UnslothTrainer internally creates optimizer like this:
# (shown for understanding, not typical usage)

def _create_unsloth_optimizer(
    model,
    optimizer_cls,
    optimizer_kwargs,
    embedding_lr=5e-5,
):
    """Create optimizer with separate embedding parameter group."""
    lr = optimizer_kwargs["lr"]
    weight_decay = optimizer_kwargs.get("weight_decay", 0.0)

    param_groups = {"non_embeddings": {}, "embeddings": {}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("modules_to_save.default.weight"):
            param_groups["embeddings"][name] = param
        else:
            param_groups["non_embeddings"][name] = param

    optimizer_grouped_parameters = [
        {"params": list(param_groups["non_embeddings"].values()),
         "weight_decay": weight_decay, "lr": lr},
        {"params": list(param_groups["embeddings"].values()),
         "weight_decay": weight_decay, "lr": embedding_lr},
    ]

    return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
</syntaxhighlight>

=== Using unsloth_train Function ===
<syntaxhighlight lang="python">
from unsloth.trainer import unsloth_train

# For older transformers versions with gradient accumulation bug
# This function provides a fixed training loop

# Note: For transformers > 4.45.2, this just calls trainer.train()
trainer_stats = unsloth_train(trainer)

# Equivalent to:
# trainer_stats = trainer.train()  # on modern transformers
</syntaxhighlight>

== Related Pages ==
=== Context & Requirements ===
* Requires TRL library (trl >= 0.7.0)
* Works with both SFTConfig and TrainingArguments
* Auto-patches for TRL version compatibility

=== Tips and Tricks ===
* Use embedding_learning_rate=5e-5 when training embed_tokens (lower than LoRA lr)
* For most cases, standard SFTTrainer with Unsloth patches is sufficient
* Enable packing=True in SFTConfig for automatic sample packing
* UnslothTrainer auto-detects modules_to_save for embedding parameters
