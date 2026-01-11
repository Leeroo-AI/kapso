# Implementation: SFTTrainer_train

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|TRL SFTTrainer|https://huggingface.co/docs/trl/sft_trainer]]
* [[source::Repo|TRL|https://github.com/huggingface/trl]]
|-
! Domains
| [[domain::NLP]], [[domain::Training]], [[domain::Supervised_Learning]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Wrapper usage pattern for TRL's SFTTrainer within Unsloth's optimized training pipeline, automatically enhanced with fused kernels and efficient gradient computation.

=== Description ===

`SFTTrainer` is TRL's supervised fine-tuning trainer. When Unsloth is imported, it patches SFTTrainer at import time to use optimized kernels for:
* Fused cross-entropy loss (chunked for large vocabularies)
* Optimized RMS normalization
* Efficient gradient checkpointing

This is a Wrapper Doc - it documents how Unsloth uses and extends TRL's SFTTrainer.

Key Unsloth enhancements:
* **Automatic patching** - Import Unsloth first to apply optimizations
* **Sample packing** - Pack multiple sequences into single training examples
* **Padding-free training** - Avoid computation on padding tokens
* **UnslothTrainer** - Custom trainer subclass with embedding LR support

=== Usage ===

Import Unsloth before TRL to apply patches. Create SFTTrainer with model, tokenizer, dataset, and training arguments. Call `trainer.train()` to execute the training loop. Unsloth optimizations are applied automatically.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/trainer.py
* '''Lines:''' L182-408 (UnslothTrainer class)

=== External Reference ===
* '''Library:''' [https://github.com/huggingface/trl TRL]
* '''Documentation:''' [https://huggingface.co/docs/trl/sft_trainer SFTTrainer Docs]

=== Signature ===
<syntaxhighlight lang="python">
# SFTTrainer is from TRL, patched by Unsloth
from trl import SFTTrainer

trainer = SFTTrainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,  # or processing_class in newer TRL
    train_dataset: Dataset,
    args: TrainingArguments,
    packing: bool = False,
    dataset_text_field: str = "text",
    max_seq_length: int = 2048,
    dataset_num_proc: int = 2,
    data_collator: Optional[DataCollator] = None,
    **kwargs,
)

# Execute training
train_output = trainer.train()
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# IMPORTANT: Import Unsloth FIRST to apply patches
from unsloth import FastLanguageModel

# Then import SFTTrainer (now patched)
from trl import SFTTrainer

# Or use UnslothTrainer for embedding_learning_rate support
from unsloth import UnslothTrainer
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
| train_dataset || Dataset || Yes || Dataset with "text" column (formatted conversations)
|-
| args || TrainingArguments || Yes || Training configuration
|-
| packing || bool || No (default: False) || Enable sample packing for efficiency
|-
| max_seq_length || int || No (default: 2048) || Maximum sequence length
|-
| dataset_text_field || str || No (default: "text") || Column containing formatted text
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| trainer || SFTTrainer || Configured trainer instance
|-
| train_output || TrainOutput || Training metrics (loss, steps, runtime)
|}

== Usage Examples ==

=== Basic SFT Training ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel, UnslothTrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

# 1. Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
)

# 2. Apply LoRA
model = FastLanguageModel.get_peft_model(model, r=16)

# 3. Load dataset (pre-formatted with chat template)
dataset = load_dataset("json", data_files="train.jsonl")["train"]

# 4. Configure training
training_args = UnslothTrainingArguments(
    output_dir = "./outputs",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    max_steps = 60,
    learning_rate = 2e-4,
    optim = "adamw_8bit",
    bf16 = True,
)

# 5. Create trainer and train
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = training_args,
    dataset_text_field = "text",
    max_seq_length = 2048,
)

trainer.train()
</syntaxhighlight>

=== With Sample Packing ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
)
model = FastLanguageModel.get_peft_model(model, r=16)

# Enable sample packing for better GPU utilization
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        output_dir = "./outputs",
        packing = True,  # Pack multiple sequences
        max_seq_length = 2048,
        per_device_train_batch_size = 2,
        bf16 = True,
    ),
)

trainer.train()
</syntaxhighlight>

=== With UnslothTrainer for Embedding LR ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
)

# Train embeddings too
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    modules_to_save = ["embed_tokens", "lm_head"],
)

training_args = UnslothTrainingArguments(
    output_dir = "./outputs",
    learning_rate = 2e-4,
    embedding_learning_rate = 5e-5,  # Lower LR for embeddings
    max_steps = 100,
    bf16 = True,
)

# Use UnslothTrainer for embedding_learning_rate support
trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = training_args,
)

trainer.train()
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_Supervised_Finetuning]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:Unslothai_Unsloth_Sample_Packing_Tip]]
