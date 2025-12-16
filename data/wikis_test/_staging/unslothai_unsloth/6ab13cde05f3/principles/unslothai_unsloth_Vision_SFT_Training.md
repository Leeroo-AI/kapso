{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Vision Training|https://docs.unsloth.ai/basics/vision-fine-tuning]]
|-
! Domains
| [[domain::Vision_Language_Models]], [[domain::Fine_Tuning]], [[domain::Supervised_Learning]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==

Supervised fine-tuning process for Vision-Language Models using specialized data collators and VLM-specific training configurations.

=== Description ===

Vision SFT training differs from text-only training:
- Smaller batch sizes (1-2) due to image memory
- Higher gradient accumulation steps
- Special data collator for image processing
- Attention mask handling for variable image sizes

== Practical Guide ==

=== Basic Vision Training ===
<syntaxhighlight lang="python">
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
import torch

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    args=SFTConfig(
        max_seq_length=2048,
        per_device_train_batch_size=1,   # Small batches for VLMs
        gradient_accumulation_steps=8,    # Compensate with accumulation
        warmup_steps=5,
        max_steps=30,
        logging_steps=1,
        output_dir="vision_outputs",
        optim="adamw_8bit",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,      # Important!
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    ),
)

trainer.train()
</syntaxhighlight>

=== Memory Optimization ===
<syntaxhighlight lang="python">
# For large VLMs on limited VRAM
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        max_seq_length=1024,  # Reduce if needed
    ),
)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_FastVisionModel]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_Vision_Language_Model_Finetuning]]
