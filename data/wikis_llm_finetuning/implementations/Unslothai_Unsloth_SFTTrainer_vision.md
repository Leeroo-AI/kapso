# Implementation: SFTTrainer_vision

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|TRL SFTTrainer|https://huggingface.co/docs/trl/sft_trainer]]
|-
! Domains
| [[domain::Computer_Vision]], [[domain::Training]], [[domain::Multimodal]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Wrapper usage pattern for SFTTrainer with Vision-Language Models, using UnslothVisionDataCollator for proper image handling.

=== Description ===

SFTTrainer for VLM training requires:
* **processing_class** instead of tokenizer (AutoProcessor)
* **UnslothVisionDataCollator** for batching images and text
* **remove_unused_columns=False** to preserve image data

This is a **Wrapper Doc** documenting the VLM-specific usage of SFTTrainer.

=== Usage ===

Create SFTTrainer with VLM model, AutoProcessor, multimodal dataset, and the vision data collator.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/trainer.py
* '''External:''' unsloth_zoo/vision_utils.py (UnslothVisionDataCollator)

=== Import ===
<syntaxhighlight lang="python">
from trl import SFTTrainer, SFTConfig
from unsloth import UnslothVisionDataCollator
</syntaxhighlight>

== Usage Examples ==

=== Complete Vision Training ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel, UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

# 1. Load VLM
model, processor = FastVisionModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
    load_in_4bit = True,
)

# 2. Apply LoRA
model = FastVisionModel.get_peft_model(
    model,
    r = 16,
    finetune_vision_layers = True,
    finetune_language_layers = True,
)

# 3. Set training mode
FastVisionModel.for_training(model)

# 4. Configure training
training_args = SFTConfig(
    output_dir = "./vision_outputs",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    max_steps = 100,
    learning_rate = 2e-4,
    bf16 = True,
    remove_unused_columns = False,  # CRITICAL for VLM
    dataset_kwargs = {"skip_prepare_dataset": True},
)

# 5. Create trainer with vision collator
trainer = SFTTrainer(
    model = model,
    processing_class = processor,  # NOT tokenizer
    train_dataset = vision_dataset,
    data_collator = UnslothVisionDataCollator(processor),
    args = training_args,
)

# 6. Train
trainer.train()
</syntaxhighlight>

=== With Sample Outputs ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel, UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

model, processor = FastVisionModel.from_pretrained(...)
model = FastVisionModel.get_peft_model(model, ...)

training_args = SFTConfig(
    output_dir = "./vision_outputs",
    per_device_train_batch_size = 1,
    max_steps = 50,
    bf16 = True,
    remove_unused_columns = False,
    logging_steps = 10,
)

trainer = SFTTrainer(
    model = model,
    processing_class = processor,
    train_dataset = vision_dataset,
    data_collator = UnslothVisionDataCollator(processor),
    args = training_args,
)

trainer.train()

# Test generation after training
FastVisionModel.for_inference(model)
# ... generate with model
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_Vision_Training]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
