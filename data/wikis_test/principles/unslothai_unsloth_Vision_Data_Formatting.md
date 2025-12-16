{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Vision Data Prep|https://docs.unsloth.ai/basics/vision-fine-tuning]]
|-
! Domains
| [[domain::Vision_Language_Models]], [[domain::Data_Preparation]], [[domain::Multimodal]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==

Process of preparing image-text paired datasets for Vision-Language Model fine-tuning with proper message formatting and image handling.

=== Description ===

Vision data formatting handles:
- Image path/URL references in messages
- Multi-modal content blocks (image + text)
- Image preprocessing and tokenization
- Attention mask handling for image tokens

== Practical Guide ==

=== Message Format ===
<syntaxhighlight lang="python">
# Standard VLM message format
sample = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/path/to/image.jpg"},
                {"type": "text", "text": "What is in this image?"}
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "This image shows..."}]
        }
    ]
}
</syntaxhighlight>

=== Dataset Preparation ===
<syntaxhighlight lang="python">
from datasets import load_dataset

dataset = load_dataset("json", data_files="vision_train.json", split="train")

def format_vision_data(examples):
    formatted = []
    for img_path, instruction, response in zip(
        examples["image"], examples["instruction"], examples["response"]
    ):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": instruction}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}]
            }
        ]
        formatted.append({"messages": messages})
    return {"formatted": formatted}

dataset = dataset.map(format_vision_data, batched=True)
</syntaxhighlight>

=== Using Vision Data Collator ===
<syntaxhighlight lang="python">
from unsloth.trainer import UnslothVisionDataCollator

data_collator = UnslothVisionDataCollator(model, tokenizer)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    data_collator=data_collator,
    args=SFTConfig(
        remove_unused_columns=False,  # Important for VLMs
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    ),
)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_FastVisionModel]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_Vision_Language_Model_Finetuning]]
