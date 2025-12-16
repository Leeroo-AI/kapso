{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Training Guide|https://docs.unsloth.ai/get-started/fine-tuning-guide]]
* [[source::Paper|TRL: Transformer Reinforcement Learning|https://arxiv.org/abs/2309.16797]]
|-
! Domains
| [[domain::Fine_Tuning]], [[domain::NLP]], [[domain::Supervised_Learning]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==

Supervised fine-tuning process using TRL's SFTTrainer with Unsloth's optimized kernels for efficient instruction tuning.

=== Description ===

SFT (Supervised Fine-Tuning) Training adapts pre-trained models to follow instructions by training on input-output pairs:

**Training Pipeline:**
1. Format data with chat templates
2. Configure SFTTrainer with hyperparameters
3. Run training with automatic optimization
4. Monitor loss and metrics

**Unsloth Optimizations:**
- Chunked cross-entropy loss for large vocabularies
- Fused LoRA operations during forward/backward
- Optimized attention with automatic backend selection
- Memory-efficient gradient checkpointing

**Key Hyperparameters:**
- Learning rate: 2e-4 to 2e-5 typical for LoRA
- Batch size: 1-8 depending on model size and VRAM
- Gradient accumulation: Scale effective batch size
- Max steps or epochs

=== Usage ===

Use SFT training when:
- Adapting models to follow specific instruction formats
- Teaching models domain-specific knowledge
- Creating chat or assistant models
- Fine-tuning on custom datasets

== Practical Guide ==

=== Basic SFT Configuration ===
<syntaxhighlight lang="python">
from trl import SFTTrainer, SFTConfig
import torch

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        # Output settings
        output_dir="outputs",

        # Training duration
        max_steps=60,              # Or use num_train_epochs=1
        logging_steps=1,
        save_steps=50,

        # Batch size
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # Effective batch = 8

        # Learning rate
        learning_rate=2e-4,
        warmup_steps=10,
        lr_scheduler_type="linear",

        # Optimizer
        optim="adamw_8bit",        # 8-bit Adam saves VRAM

        # Precision
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),

        # Sequence settings
        max_seq_length=2048,
    ),
)

trainer.train()
</syntaxhighlight>

=== With Evaluation ===
<syntaxhighlight lang="python">
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=SFTConfig(
        output_dir="outputs",
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        # ... other settings
    ),
)
</syntaxhighlight>

=== With Sequence Packing ===
<syntaxhighlight lang="python">
# Packing combines multiple short sequences into one
# More efficient use of max_seq_length
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="outputs",
        packing=True,  # Enable sequence packing
        max_seq_length=2048,
        # ... other settings
    ),
)
</syntaxhighlight>

=== Monitoring Training ===
<syntaxhighlight lang="python">
# With Weights & Biases
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="outputs",
        report_to="wandb",
        run_name="my-qlora-run",
        # ... other settings
    ),
)

# Or with TensorBoard
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="outputs",
        report_to="tensorboard",
        logging_dir="./logs",
        # ... other settings
    ),
)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_FastLanguageModel]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_QLoRA_Finetuning]]
