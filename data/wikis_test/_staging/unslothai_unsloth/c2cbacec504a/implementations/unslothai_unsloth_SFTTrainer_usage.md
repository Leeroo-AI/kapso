# Implementation: unslothai_unsloth_SFTTrainer_usage

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|TRL SFTTrainer|https://huggingface.co/docs/trl/sft_trainer]]
* [[source::Doc|Transformers TrainingArguments|https://huggingface.co/docs/transformers/main_classes/trainer]]
|-
! Domains
| [[domain::NLP]], [[domain::Training]], [[domain::Supervised_Learning]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Wrapper documentation for using TRL's SFTTrainer with Unsloth optimizations for supervised fine-tuning.

=== Description ===

SFTTrainer from the TRL library is the standard trainer for supervised fine-tuning of language models. When Unsloth is imported first, it patches SFTTrainer to use:

1. **Fused cross-entropy loss**: Triton kernel that's 2x faster than PyTorch
2. **Optimized gradient checkpointing**: Selective checkpointing that balances speed and memory
3. **Padding-free training**: Packs sequences to eliminate wasted compute on padding tokens
4. **Sample packing**: Concatenates short sequences into longer ones for efficiency

This wrapper documents Unsloth-specific configuration options and best practices.

=== Usage ===

Use SFTTrainer after:
1. Loading model with `FastLanguageModel.from_pretrained`
2. Injecting LoRA with `FastLanguageModel.get_peft_model`
3. Formatting dataset with chat templates

This is the primary training interface for supervised fine-tuning workflows.

== Code Reference ==

=== Source Location ===
* '''External Library:''' [https://github.com/huggingface/trl TRL]
* '''Unsloth Patches:''' unsloth/trainer.py (L1-437)

=== External Documentation ===
* [https://huggingface.co/docs/trl/sft_trainer TRL SFTTrainer Documentation]

=== Signature ===
<syntaxhighlight lang="python">
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    args: SFTConfig,
    eval_dataset: Optional[Dataset] = None,
    data_collator: Optional[DataCollator] = None,
    packing: bool = False,
    dataset_text_field: str = "text",
    max_seq_length: Optional[int] = None,
    dataset_num_proc: Optional[int] = None,
    formatting_func: Optional[Callable] = None,
)
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# IMPORTANT: Import unsloth first to apply patches
import unsloth
from unsloth import FastLanguageModel

# Then import TRL (now patched)
from trl import SFTTrainer, SFTConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || PreTrainedModel || Yes || Model with LoRA adapters from get_peft_model
|-
| tokenizer || PreTrainedTokenizer || Yes || Tokenizer with chat template configured
|-
| train_dataset || Dataset || Yes || Formatted training dataset with "text" column
|-
| args || SFTConfig || Yes || Training configuration (batch size, learning rate, etc.)
|-
| packing || bool || No (default: False) || Enable sequence packing for efficiency
|-
| max_seq_length || int || No || Maximum sequence length (uses model's if not set)
|-
| formatting_func || Callable || No || Custom formatting function for dataset
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| trainer || SFTTrainer || Configured trainer ready for `.train()` call
|}

== Usage Examples ==

=== Basic SFT Setup ===
<syntaxhighlight lang="python">
import unsloth
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# 1. Load and configure model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model, r = 16, lora_alpha = 16,
)

# 2. Load and format dataset
dataset = load_dataset("your_dataset", split="train")
# Assume dataset has "text" column with formatted conversations

# 3. Configure training
training_args = SFTConfig(
    output_dir = "./outputs",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,  # Effective batch = 8
    warmup_steps = 5,
    max_steps = 100,
    learning_rate = 2e-4,
    logging_steps = 1,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 3407,
    report_to = "none",
)

# 4. Create trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = training_args,
    max_seq_length = 2048,
    dataset_text_field = "text",
)

# 5. Train
trainer.train()
</syntaxhighlight>

=== With Sequence Packing ===
<syntaxhighlight lang="python">
# Enable packing for datasets with many short sequences
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = training_args,
    packing = True,  # Pack sequences together
    max_seq_length = 2048,
    dataset_text_field = "text",
)
</syntaxhighlight>

=== With Custom Formatting Function ===
<syntaxhighlight lang="python">
# Define custom formatting
def formatting_prompts_func(examples):
    texts = []
    for instruction, output in zip(examples["instruction"], examples["output"]):
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        texts.append(text)
    return {"text": texts}

# Use formatting function instead of pre-formatted text
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = training_args,
    formatting_func = formatting_prompts_func,
    max_seq_length = 2048,
)
</syntaxhighlight>

=== Recommended Unsloth Settings ===
<syntaxhighlight lang="python">
# Optimized settings for Unsloth
training_args = SFTConfig(
    output_dir = "./outputs",

    # Batch size settings
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,

    # Optimizer settings (8-bit Adam saves memory)
    optim = "adamw_8bit",
    learning_rate = 2e-4,
    weight_decay = 0.01,
    lr_scheduler_type = "linear",

    # Warmup
    warmup_ratio = 0.03,

    # Logging
    logging_steps = 1,
    report_to = "none",  # or "wandb", "tensorboard"

    # Saving
    save_strategy = "steps",
    save_steps = 100,

    # Reproducibility
    seed = 3407,

    # Mixed precision (auto-detected by Unsloth)
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
)
</syntaxhighlight>

== Unsloth-Specific Optimizations ==

| Optimization | Effect | Enabled By |
|--------------|--------|------------|
| Fused cross-entropy | 2x faster loss computation | Automatic when `import unsloth` |
| Smart gradient checkpointing | 30% less memory | `use_gradient_checkpointing="unsloth"` |
| Padding-free training | No wasted compute on padding | Automatic |
| 8-bit Adam | ~25% memory savings | `optim="adamw_8bit"` |

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:unslothai_unsloth_Training_Configuration]]

=== Requires Environment ===
* [[requires_env::Environment:unslothai_unsloth_CUDA]]
