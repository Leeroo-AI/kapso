# Implementation: UnslothVisionDataCollator

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Repo|unsloth-zoo|https://github.com/unslothai/unsloth-zoo]]
|-
! Domains
| [[domain::VLMs]], [[domain::Training]], [[domain::Data_Processing]], [[domain::Multimodal]]
|-
! Last Updated
| [[last_updated::2025-12-15 19:00 GMT]]
|}

== Overview ==
Concrete tool for collating vision-language training batches with proper image handling provided by the Unsloth library.

=== Description ===
`UnslothVisionDataCollator` is a specialized data collator for training vision-language models (VLMs). It handles the complexity of batching multimodal data containing both images and text:

1. **Message Formatting** - Converts conversation-style messages with images to model inputs
2. **Image Processing** - Handles PIL images, URLs, and base64-encoded images
3. **Tensor Batching** - Properly batches pixel values alongside text tokens
4. **Attention Masking** - Creates correct attention masks for variable-length sequences

The collator works with VLMs like Qwen2-VL, Llama 3.2 Vision, Pixtral, and other multimodal models supported by Unsloth.

=== Usage ===
Use this collator when:
- Training VLMs with SFTTrainer on image-text datasets
- Working with conversation-format data containing images
- Need proper multimodal batch creation

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth-zoo unslothai/unsloth-zoo]
* '''File:''' unsloth_zoo/vision_utils.py
* '''Re-exported from:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/trainer.py#L36-L37 unsloth/trainer.py]

Source Files: unsloth_zoo/vision_utils.py; unsloth/trainer.py:L36-L37

=== Signature ===
<syntaxhighlight lang="python">
class UnslothVisionDataCollator:
    """Data collator for vision-language model training."""

    def __init__(
        self,
        model: PreTrainedModel,
        processor: Union[PreTrainedTokenizer, ProcessorMixin],
    ):
        """
        Initialize the vision data collator.

        Args:
            model: The vision-language model being trained
            processor: Tokenizer or processor for the model
        """

    def __call__(
        self,
        examples: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of vision-language examples.

        Args:
            examples: List of dictionaries with 'messages' key containing
                     conversation-format data with text and images

        Returns:
            Dictionary with:
            - input_ids: Tokenized text input
            - attention_mask: Attention mask
            - labels: Training labels (-100 for masked tokens)
            - pixel_values: Processed image tensors (if images present)
            - image_sizes: Original image dimensions (for some models)
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth.trainer import UnslothVisionDataCollator

# Or from unsloth_zoo directly:
from unsloth_zoo.vision_utils import UnslothVisionDataCollator
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || PreTrainedModel || Yes || VLM being trained
|-
| processor || PreTrainedTokenizer/Processor || Yes || Model's tokenizer/processor
|-
| examples || List[Dict] || Yes || Batch of conversation examples
|}

=== Input Data Format ===
Each example should have a "messages" key with conversation format:

<syntaxhighlight lang="python">
{
    "messages": [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image", "image": <PIL.Image or URL>}
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "This image shows..."}]
        }
    ]
}
</syntaxhighlight>

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| input_ids || torch.Tensor || Tokenized text [batch, seq_len]
|-
| attention_mask || torch.Tensor || Attention mask [batch, seq_len]
|-
| labels || torch.Tensor || Training labels (user turns masked with -100)
|-
| pixel_values || torch.Tensor || Processed images [batch, channels, H, W]
|-
| image_sizes || torch.Tensor || Original image sizes (model-dependent)
|}

== Usage Examples ==

=== Basic VLM Training Setup ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

# 1. Load VLM
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2-VL-2B-Instruct",
    load_in_4bit=True,
)

# 2. Add LoRA adapters
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    r=16,
)

# 3. Create data collator
data_collator = UnslothVisionDataCollator(model, tokenizer)

# 4. Setup trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=train_dataset,
    args=SFTConfig(
        output_dir="vlm_output",
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        max_steps=100,
        remove_unused_columns=False,  # IMPORTANT for multimodal
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    ),
)

trainer.train()
</syntaxhighlight>

=== Preparing Dataset ===
<syntaxhighlight lang="python">
from datasets import load_dataset
from PIL import Image

# Load an OCR or image captioning dataset
raw_dataset = load_dataset("your-vlm-dataset")

def convert_to_messages(sample):
    """Convert dataset sample to message format."""
    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful vision assistant."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sample["question"]},
                    {"type": "image", "image": sample["image"]},  # PIL Image
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["answer"]}]
            }
        ]
    }

train_dataset = [convert_to_messages(s) for s in raw_dataset]
</syntaxhighlight>

=== With Image URLs ===
<syntaxhighlight lang="python">
# Images can be URLs (will be downloaded automatically)
example = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image:"},
                {"type": "image", "image": "https://example.com/image.jpg"},
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "The image shows..."}]
        }
    ]
}
</syntaxhighlight>

=== With Multiple Images ===
<syntaxhighlight lang="python">
# Some models support multiple images per turn
example = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare these two images:"},
                {"type": "image", "image": image1},  # PIL Image
                {"type": "image", "image": image2},  # PIL Image
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "The first image shows... while the second shows..."}]
        }
    ]
}
</syntaxhighlight>

=== Custom Data Processing ===
<syntaxhighlight lang="python">
import torch
from unsloth.trainer import UnslothVisionDataCollator

# Create collator
collator = UnslothVisionDataCollator(model, tokenizer)

# Manually collate a batch
batch = [example1, example2, example3]
collated = collator(batch)

# Access collated data
print(f"input_ids shape: {collated['input_ids'].shape}")
print(f"attention_mask shape: {collated['attention_mask'].shape}")
print(f"labels shape: {collated['labels'].shape}")
if 'pixel_values' in collated:
    print(f"pixel_values shape: {collated['pixel_values'].shape}")
</syntaxhighlight>

=== Full Training Example ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from PIL import Image

# Load model
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2-VL-2B-Instruct",
    load_in_4bit=True,
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    r=16,
    lora_alpha=32,
)

# Prepare dataset (example with DocVQA)
def format_docvqa(sample):
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": sample["query"]},
                {"type": "image", "image": sample["image"]},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": sample["answers"][0]},
            ]},
        ]
    }

dataset = load_dataset("HuggingFaceM4/DocumentVQA", split="train[:1000]")
train_data = [format_docvqa(s) for s in dataset]

# Enable training mode
FastVisionModel.for_training(model)

# Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=train_data,
    args=SFTConfig(
        output_dir="docvqa_output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=500,
        fp16=True,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    ),
)

trainer.train()

# Save
model.save_pretrained("docvqa_lora")
</syntaxhighlight>

== Related Pages ==
=== Context & Requirements ===
* Requires vision-language model from FastVisionModel
* Works with PIL Images, URLs, or base64-encoded images
* Requires `remove_unused_columns=False` in SFTConfig

=== Tips and Tricks ===
* Always set `remove_unused_columns=False` for multimodal training
* Use `dataset_kwargs={"skip_prepare_dataset": True}` with SFTTrainer
* Images are automatically resized based on model requirements
* For text-only samples, omit the image entry in content list
