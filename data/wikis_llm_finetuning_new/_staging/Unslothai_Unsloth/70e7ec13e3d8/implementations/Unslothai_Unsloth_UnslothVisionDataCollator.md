# Implementation: UnslothVisionDataCollator

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|HuggingFace Data Collators|https://huggingface.co/docs/transformers/main_classes/data_collator]]
|-
! Domains
| [[domain::Computer_Vision]], [[domain::Training]], [[domain::Data_Collation]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

Concrete tool for collating multimodal batches with images and text provided by Unsloth.

=== Description ===

`UnslothVisionDataCollator` is a specialized data collator for vision-language model training. It handles the complexity of:

* Processing conversation messages with embedded images
* Managing variable-length sequences due to different image sizes
* Creating attention masks that properly handle image tokens
* Batching samples with different numbers of images

The collator is imported from `unsloth_zoo.vision_utils` and is required when using SFTTrainer with vision models.

=== Usage ===

Create an instance of UnslothVisionDataCollator with the processor from your vision model. Pass it to SFTTrainer's `data_collator` parameter. Also set `remove_unused_columns=False` and `dataset_text_field=""` in SFTConfig.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/trainer.py
* '''Lines:''' 36-38 (import from unsloth_zoo.vision_utils)

=== Signature ===
<syntaxhighlight lang="python">
class UnslothVisionDataCollator:
    def __init__(
        self,
        processor: AutoProcessor,
    ):
        """
        Data collator for vision-language model training.

        Args:
            processor: AutoProcessor from FastVisionModel.from_pretrained
        """

    def __call__(
        self,
        features: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of multimodal features.

        Args:
            features: List of sample dicts with "messages" key

        Returns:
            Batched tensors: input_ids, attention_mask, pixel_values, labels
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth import UnslothVisionDataCollator

# Or from the trainer module
from unsloth.trainer import UnslothVisionDataCollator
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| processor || AutoProcessor || Yes || Processor from FastVisionModel.from_pretrained
|}

=== Collator Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| features || List[Dict] || Yes || List of samples with "messages" key containing conversation format
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| input_ids || torch.Tensor || Tokenized input sequences [batch, seq_len]
|-
| attention_mask || torch.Tensor || Attention mask [batch, seq_len]
|-
| pixel_values || torch.Tensor || Processed images [batch, channels, height, width]
|-
| labels || torch.Tensor || Labels for loss computation [batch, seq_len]
|}

== Usage Examples ==

=== Basic Vision Training Setup ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel, UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# Load model
model, processor = FastVisionModel.from_pretrained(
    model_name="unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastVisionModel.get_peft_model(
    model,
    r=16,
    finetune_vision_layers=True,
    finetune_language_layers=True,
)

# Prepare dataset (must have "messages" column)
dataset = Dataset.from_list([...])

# Create data collator
data_collator = UnslothVisionDataCollator(processor)

# Configure training - NOTE the vision-specific settings
training_args = SFTConfig(
    output_dir="./vision_output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    bf16=True,
    logging_steps=10,
    remove_unused_columns=False,  # REQUIRED for vision
    dataset_text_field="",         # REQUIRED for vision
)

# Create trainer with data collator
trainer = SFTTrainer(
    model=model,
    processing_class=processor,  # Note: processor, not tokenizer
    train_dataset=dataset,
    args=training_args,
    data_collator=data_collator,  # REQUIRED for vision
)

# Train
trainer.train()
</syntaxhighlight>

=== Complete OCR Fine-Tuning Example ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel, UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from PIL import Image

# 1. Load model
model, processor = FastVisionModel.from_pretrained(
    model_name="unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# 2. Add LoRA
model = FastVisionModel.get_peft_model(
    model,
    r=16,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    use_gradient_checkpointing="unsloth",
)

# 3. Prepare dataset
def format_ocr_data(example):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open(example["image_path"])},
                {"type": "text", "text": example["question"]},
            ],
        },
        {
            "role": "assistant",
            "content": example["answer"],
        },
    ]
    return {"messages": messages}

raw_data = [
    {"image_path": "doc1.png", "question": "OCR this document", "answer": "..."},
]
dataset = Dataset.from_list(raw_data).map(format_ocr_data)

# 4. Create collator and config
data_collator = UnslothVisionDataCollator(processor)

training_args = SFTConfig(
    output_dir="./ocr_output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    bf16=True,
    remove_unused_columns=False,
    dataset_text_field="",
    save_steps=100,
    logging_steps=10,
)

# 5. Train
trainer = SFTTrainer(
    model=model,
    processing_class=processor,
    train_dataset=dataset,
    args=training_args,
    data_collator=data_collator,
)

trainer.train()

# 6. Save
model.save_pretrained_merged("./ocr_model", processor, save_method="merged_16bit")
</syntaxhighlight>

== Common Issues ==

=== "remove_unused_columns must be False" ===
<syntaxhighlight lang="python">
# Wrong
training_args = SFTConfig(
    output_dir="./output",
    # remove_unused_columns defaults to True
)

# Correct
training_args = SFTConfig(
    output_dir="./output",
    remove_unused_columns=False,  # Required!
)
</syntaxhighlight>

=== Empty dataset_text_field ===
<syntaxhighlight lang="python">
# Wrong - expects "text" column
training_args = SFTConfig(
    output_dir="./output",
    dataset_text_field="text",
)

# Correct - vision uses messages column processed by collator
training_args = SFTConfig(
    output_dir="./output",
    dataset_text_field="",  # Required!
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_Vision_Training_Setup]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_TRL]]
* [[requires_env::Environment:Unslothai_Unsloth_Vision]]
