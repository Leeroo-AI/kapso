# Implementation: Multimodal_Data_Preparation_Pattern

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|HuggingFace Datasets|https://huggingface.co/docs/datasets]]
|-
! Domains
| [[domain::Computer_Vision]], [[domain::NLP]], [[domain::Data_Preprocessing]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

Pattern documentation for preparing multimodal datasets for vision-language model training.

=== Description ===

This is a **Pattern Doc** describing the user-defined data preparation interface for vision model training. Users must format their data according to the conversation schema with embedded image references.

=== Usage ===

Follow this pattern when preparing any image-text dataset for VLM fine-tuning. The exact implementation depends on your data source, but the output must match the expected message format.

== Interface Specification ==

=== Message Format ===
<syntaxhighlight lang="python">
# Required schema for vision model training
{
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": <PIL.Image>},
                {"type": "text", "text": "Question about the image"},
            ],
        },
        {
            "role": "assistant",
            "content": "Response text",  # String, not list
        },
    ]
}
</syntaxhighlight>

=== Image Sources ===
<syntaxhighlight lang="python">
# Images can be provided as:
from PIL import Image

# 1. PIL Image object
{"type": "image", "image": Image.open("path/to/image.jpg")}

# 2. Local file path
{"type": "image", "image": "/path/to/image.jpg"}

# 3. URL (will be downloaded)
{"type": "image", "image": "https://example.com/image.jpg"}

# 4. Base64 encoded string
{"type": "image", "image": "data:image/jpeg;base64,/9j/4AAQ..."}
</syntaxhighlight>

== Example Implementations ==

=== OCR Dataset Preparation ===
<syntaxhighlight lang="python">
from PIL import Image
from datasets import Dataset
from unsloth import FastVisionModel

# Load model and processor
model, processor = FastVisionModel.from_pretrained(
    model_name="unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Raw OCR data
raw_data = [
    {
        "image_path": "receipts/receipt_001.png",
        "question": "What is the total amount?",
        "answer": "$42.50",
    },
    {
        "image_path": "receipts/receipt_002.png",
        "question": "List all items purchased.",
        "answer": "Coffee ($4.50), Sandwich ($8.00), Cookie ($2.00)",
    },
]

def format_ocr_example(example):
    """Format a single OCR example for VLM training."""
    # Load image
    image = Image.open(example["image_path"]).convert("RGB")

    # Create conversation
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": example["question"]},
            ],
        },
        {
            "role": "assistant",
            "content": example["answer"],
        },
    ]

    return {"messages": messages}

# Create dataset
dataset = Dataset.from_list(raw_data)
dataset = dataset.map(format_ocr_example)

print(dataset[0]["messages"])
</syntaxhighlight>

=== Multi-Image Conversation ===
<syntaxhighlight lang="python">
from PIL import Image
from datasets import Dataset

def format_multi_image(example):
    """Format an example with multiple images."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open(example["image1"])},
                {"type": "image", "image": Image.open(example["image2"])},
                {"type": "text", "text": "Compare these two images."},
            ],
        },
        {
            "role": "assistant",
            "content": example["comparison"],
        },
    ]
    return {"messages": messages}

# Example with two images per sample
data = [
    {
        "image1": "cat.jpg",
        "image2": "dog.jpg",
        "comparison": "The first image shows a cat, while the second shows a dog.",
    },
]

dataset = Dataset.from_list(data)
dataset = dataset.map(format_multi_image)
</syntaxhighlight>

=== Document Understanding Dataset ===
<syntaxhighlight lang="python">
from PIL import Image
from datasets import Dataset, load_dataset
import json

# Load a document understanding dataset
def prepare_docvqa(example):
    """Prepare DocVQA-style dataset."""
    # Load image
    image = example["image"]  # Already a PIL Image in HF datasets
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": example["question"]},
            ],
        },
        {
            "role": "assistant",
            "content": example["answers"][0],  # Take first answer
        },
    ]

    return {"messages": messages}

# Load and prepare
raw_dataset = load_dataset("HuggingFaceM4/DocumentVQA", split="train[:1000]")
dataset = raw_dataset.map(prepare_docvqa)
</syntaxhighlight>

=== System Prompt with Images ===
<syntaxhighlight lang="python">
def format_with_system_prompt(example):
    """Include a system prompt in vision conversation."""
    messages = [
        {
            "role": "system",
            "content": "You are an expert document analyst. Extract information accurately.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open(example["image_path"])},
                {"type": "text", "text": example["query"]},
            ],
        },
        {
            "role": "assistant",
            "content": example["response"],
        },
    ]
    return {"messages": messages}
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| raw_data || List[Dict] or Dataset || Yes || Raw data with images and text
|-
| processor || AutoProcessor || Yes || Processor from FastVisionModel.from_pretrained
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| dataset || Dataset || Dataset with "messages" column in VLM format
|}

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_Multimodal_Data_Preparation]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_Vision]]
