# Implementation: multimodal_dataset_pattern

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|HuggingFace Datasets|https://huggingface.co/docs/datasets]]
|-
! Domains
| [[domain::Computer_Vision]], [[domain::Data_Engineering]], [[domain::Multimodal]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Pattern specification for preparing datasets for Vision-Language Model fine-tuning with interleaved image and text content.

=== Description ===

This is a **Pattern Doc** - it documents the dataset format expected by VLM training. Multimodal datasets must provide:
* Messages with `{"type": "image"}` and `{"type": "text"}` content
* Images as PIL.Image objects or file paths
* Proper message structure for chat templates

=== Usage ===

Prepare your dataset to match this format before passing to SFTTrainer with a VLM.

== Interface Specification ==

=== Required Format ===
<syntaxhighlight lang="python">
# Each dataset row should have:
{
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image"},  # Placeholder for image
                {"type": "text", "text": "What is in this image?"}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "This image shows..."}
            ]
        }
    ],
    "images": [PIL.Image or path]  # Actual image data
}
</syntaxhighlight>

== Usage Examples ==

=== Image Captioning Dataset ===
<syntaxhighlight lang="python">
from datasets import Dataset
from PIL import Image

def create_vlm_dataset(image_paths, captions):
    """Create dataset for VLM fine-tuning."""
    data = []
    for img_path, caption in zip(image_paths, captions):
        data.append({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe this image in detail."}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": caption}
                    ]
                }
            ],
            "images": [Image.open(img_path)]
        })
    return Dataset.from_list(data)
</syntaxhighlight>

=== VQA Dataset ===
<syntaxhighlight lang="python">
from datasets import Dataset
from PIL import Image

def create_vqa_dataset(examples):
    """Create VQA dataset with questions and answers."""
    data = []
    for ex in examples:
        data.append({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": ex["question"]}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ex["answer"]}
                    ]
                }
            ],
            "images": [Image.open(ex["image_path"])]
        })
    return Dataset.from_list(data)
</syntaxhighlight>

=== Multi-Turn Conversation ===
<syntaxhighlight lang="python">
from datasets import Dataset
from PIL import Image

def create_conversation_dataset(conversations):
    """Create multi-turn conversation dataset."""
    data = []
    for conv in conversations:
        messages = []
        for turn in conv["turns"]:
            content = []
            if turn.get("has_image"):
                content.append({"type": "image"})
            content.append({"type": "text", "text": turn["text"]})

            messages.append({
                "role": turn["role"],
                "content": content
            })

        data.append({
            "messages": messages,
            "images": [Image.open(p) for p in conv.get("image_paths", [])]
        })
    return Dataset.from_list(data)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_Multimodal_Data_Preparation]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
