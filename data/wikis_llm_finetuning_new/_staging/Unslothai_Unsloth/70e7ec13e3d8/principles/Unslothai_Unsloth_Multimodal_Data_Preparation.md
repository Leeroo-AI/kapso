# Principle: Multimodal_Data_Preparation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|LLaVA: Large Language and Vision Assistant|https://arxiv.org/abs/2304.08485]]
* [[source::Doc|HuggingFace Multimodal|https://huggingface.co/docs/transformers/tasks/image_text_to_text]]
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Computer_Vision]], [[domain::NLP]], [[domain::Data_Preprocessing]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

Mechanism for preparing datasets containing images and text in the conversation format required by vision-language models.

=== Description ===

Multimodal data preparation structures image-text pairs into a conversation format where messages can contain both image references and text content. This format enables:

1. **Image Embedding**: Images are referenced within message content
2. **Interleaved Content**: Text and images can be interleaved in a single turn
3. **Multi-Turn Conversations**: Multiple exchanges with potentially multiple images

The processor handles:
* Image loading and preprocessing (resizing, normalization)
* Text tokenization
* Creating attention masks that properly handle image tokens
* Padding and batching across variable-length sequences

=== Usage ===

Use this principle when:
* Preparing datasets for vision model fine-tuning
* Working with OCR, VQA, image captioning, or document understanding tasks
* Images need to be embedded within conversational context
* Data includes multiple images per conversation

This step follows LoRA configuration and precedes training setup.

== Theoretical Basis ==

'''Message Format for VLMs:'''
<syntaxhighlight lang="python">
# Standard message format with multimodal content
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": <PIL.Image or path>},
            {"type": "text", "text": "What does this document say?"},
        ],
    },
    {
        "role": "assistant",
        "content": "The document contains...",
    },
]
</syntaxhighlight>

'''Image Token Insertion:'''
When processed, images become special tokens in the sequence:

<syntaxhighlight lang="text">
[BOS] <|image_pad|> <|image_pad|> ... [text tokens] [EOS]
       ^-- Variable number of image tokens based on resolution
</syntaxhighlight>

'''Dataset Schema:'''
{| class="wikitable"
|-
! Field !! Type !! Description
|-
| messages || List[Dict] || Conversation turns with role and content
|-
| content || List[Dict] || For user, can contain {"type": "image"} and {"type": "text"}
|-
| image || PIL.Image/str/URL || Image data, path, or URL
|}

== Practical Guide ==

=== Step 1: Load Images ===
<syntaxhighlight lang="python">
from PIL import Image
from datasets import Dataset

# Images can be PIL objects, paths, or URLs
data = [
    {
        "image": Image.open("document.png"),
        "question": "What is the total on this receipt?",
        "answer": "The total is $42.50",
    },
]
</syntaxhighlight>

=== Step 2: Format as Conversations ===
<syntaxhighlight lang="python">
def format_for_vision(example):
    """Convert to VLM conversation format."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example["image"]},
                {"type": "text", "text": example["question"]},
            ],
        },
        {
            "role": "assistant",
            "content": example["answer"],
        },
    ]
    return {"messages": messages}

dataset = Dataset.from_list(data)
dataset = dataset.map(format_for_vision)
</syntaxhighlight>

=== Step 3: Apply Processor ===
<syntaxhighlight lang="python">
# Processor handles both images and text
text = processor.apply_chat_template(
    example["messages"],
    tokenize=False,
    add_generation_prompt=False,
)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_Multimodal_Data_Preparation_Pattern]]

