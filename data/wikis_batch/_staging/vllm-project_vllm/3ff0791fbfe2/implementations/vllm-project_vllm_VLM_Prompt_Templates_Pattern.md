= VLM Prompt Templates Pattern =

{{Metadata
| Knowledge Sources = examples/offline_inference/vision_language.py, model HuggingFace pages
| Domains = Prompt Engineering, Template Design, Model-Specific Patterns
| Last Updated = 2025-12-17
}}

== Overview ==

The '''VLM Prompt Templates Pattern''' implements the Multimodal Prompt Formatting Principle by documenting and demonstrating the correct prompt structure for various vision-language models supported by vLLM. This pattern provides concrete examples of how to format prompts for different VLM architectures.

== Code Reference ==

=== Source Location ===

<syntaxhighlight lang="python">
# Examples for all supported models
examples/offline_inference/vision_language.py: model_example_map

# Processing utilities
vllm/multimodal/processing.py: MultiModalProcessor
vllm/inputs/preprocess.py: InputPreprocessor
</syntaxhighlight>

=== Common Pattern ===

<syntaxhighlight lang="python">
# General structure
prompt_dict = {
    "prompt": "<MODEL_SPECIFIC_TEMPLATE>",
    "multi_modal_data": {"image": image_object}
}

# The template contains:
# 1. Chat markers (e.g., USER:, ASSISTANT:, <|im_start|>)
# 2. Image placeholder (e.g., <image>, <IMG>, <|image_pad|>)
# 3. User instruction/question
</syntaxhighlight>

== Description ==

VLM prompt templates vary significantly across model families. Key template components include:

* '''LLaVA-style''': `USER: <image>\n{question}\nASSISTANT:`
* '''Qwen-style''': `<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{question}<|im_end|>`
* '''Phi-style''': `<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n`
* '''Mistral-style''': `[INST]{question}\n[IMG][/INST]`
* '''InternVL-style''': Chat template with `<image>` placeholder

The pattern implementation includes validation to ensure placeholders match the provided multimodal data.

== I/O Contract ==

=== Input Components ===

{| class="wikitable"
! Component !! Description !! Example
|-
| Chat markers || Role indicators || USER:, ASSISTANT:, <|im_start|>
|-
| Image placeholder || Visual feature location || <image>, <IMG>, <|image_pad|>
|-
| Question/instruction || User's query || "What is in this image?"
|-
| Special tokens || Model-specific markers || <|im_end|>, <|vision_start|>
|}

=== Output ===

A properly formatted string that the model can parse to locate image features and understand the instruction.

== Usage Examples ==

=== LLaVA-1.5 Template ===

<syntaxhighlight lang="python">
from PIL import Image
from vllm import LLM, SamplingParams

llm = LLM(model="llava-hf/llava-1.5-7b-hf")
image = Image.open("image.jpg")

# LLaVA uses simple USER:/ASSISTANT: format with <image> placeholder
prompt = {
    "prompt": "USER: <image>\nWhat is in this image?\nASSISTANT:",
    "multi_modal_data": {"image": image}
}

outputs = llm.generate([prompt], SamplingParams(max_tokens=100))
</syntaxhighlight>

=== Qwen2-VL Template ===

<syntaxhighlight lang="python">
from PIL import Image
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2-VL-7B-Instruct")
image = Image.open("image.jpg")

# Qwen2-VL uses complex template with vision markers
prompt = {
    "prompt": (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "What is in this image?<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "multi_modal_data": {"image": image}
}

outputs = llm.generate([prompt], SamplingParams(max_tokens=100))
</syntaxhighlight>

=== Phi-3-Vision Template ===

<syntaxhighlight lang="python">
from PIL import Image
from vllm import LLM, SamplingParams

llm = LLM(
    model="microsoft/Phi-3.5-vision-instruct",
    trust_remote_code=True,
    mm_processor_kwargs={"num_crops": 16}
)
image = Image.open("image.jpg")

# Phi-3-Vision uses numbered image placeholders
prompt = {
    "prompt": "<|user|>\n<|image_1|>\nDescribe this image.<|end|>\n<|assistant|>\n",
    "multi_modal_data": {"image": image}
}

outputs = llm.generate([prompt], SamplingParams(max_tokens=100))
</syntaxhighlight>

=== InternVL with Chat Template ===

<syntaxhighlight lang="python">
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_name = "OpenGVLab/InternVL3-2B"
llm = LLM(model=model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
image = Image.open("image.jpg")

# Use tokenizer's chat template
messages = [[{"role": "user", "content": "<image>\nWhat is in this image?"}]]
prompt_str = tokenizer.apply_chat_template(
    messages[0], tokenize=False, add_generation_prompt=True
)

prompt = {
    "prompt": prompt_str,
    "multi_modal_data": {"image": image}
}

outputs = llm.generate([prompt], SamplingParams(max_tokens=100))
</syntaxhighlight>

=== Pixtral (Mistral-style) Template ===

<syntaxhighlight lang="python">
from PIL import Image
from vllm import LLM, SamplingParams

llm = LLM(model="mistral-community/pixtral-12b")
image = Image.open("image.jpg")

# Pixtral uses Mistral instruction format
prompt = {
    "prompt": "<s>[INST]Describe this image in detail.\n[IMG][/INST]",
    "multi_modal_data": {"image": image}
}

outputs = llm.generate([prompt], SamplingParams(max_tokens=100))
</syntaxhighlight>

=== Multi-Image Template (InternVL) ===

<syntaxhighlight lang="python">
from PIL import Image
from vllm import LLM, SamplingParams

llm = LLM(
    model="OpenGVLab/InternVL3-2B",
    trust_remote_code=True,
    limit_mm_per_prompt={"image": 2}
)

image1 = Image.open("image1.jpg")
image2 = Image.open("image2.jpg")

# Multiple placeholders for multiple images
prompt = {
    "prompt": "<image><image>\nCompare these two images. What are the differences?",
    "multi_modal_data": {"image": [image1, image2]}
}

outputs = llm.generate([prompt], SamplingParams(max_tokens=150))
</syntaxhighlight>

=== Video Template (LLaVA-NeXT-Video) ===

<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.assets.video import VideoAsset

llm = LLM(model="llava-hf/LLaVA-NeXT-Video-7B-hf")
video = VideoAsset(name="baby_reading", num_frames=16).np_ndarrays

# Video uses <video> placeholder
prompt = {
    "prompt": "USER: <video>\nWhy is this video funny? ASSISTANT:",
    "multi_modal_data": {"video": video}
}

outputs = llm.generate([prompt], SamplingParams(max_tokens=100))
</syntaxhighlight>

=== Template with System Prompt ===

<syntaxhighlight lang="python">
from PIL import Image
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2-VL-7B-Instruct")
image = Image.open("image.jpg")

# Include system prompt for behavior control
prompt = {
    "prompt": (
        "<|im_start|>system\n"
        "You are a helpful assistant that provides detailed image descriptions.<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        "Describe this image in detail.<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "multi_modal_data": {"image": image}
}

outputs = llm.generate([prompt], SamplingParams(temperature=0.7, max_tokens=200))
</syntaxhighlight>

== Model-Specific Notes ==

=== LLaVA Family ===
* Simple template with `<image>` placeholder
* Image typically placed before question
* Uses USER:/ASSISTANT: role markers

=== Qwen Family ===
* Complex template with vision markers
* Uses `<|image_pad|>` or `<|video_pad|>` placeholders
* Requires `<|vision_start|>` and `<|vision_end|>` markers

=== Phi Family ===
* Numbered image placeholders: `<|image_1|>`, `<|image_2|>`
* Uses `<|user|>`, `<|end|>`, `<|assistant|>` markers
* `num_crops` processor kwarg affects token count

=== InternVL Family ===
* Uses chat template from tokenizer
* Simple `<image>` or `<video>` placeholder
* Apply chat template with `apply_chat_template()`

=== Mistral/Pixtral Family ===
* Uses `[INST]...[/INST]` instruction format
* `[IMG]` placeholder typically at end
* Mistral tokenizer handles special formatting

== Related Pages ==

* [[implements::Principle:vllm-project_vllm_Multimodal_Prompt_Formatting_Principle]]
* [[next_step::vllm-project_vllm_LLM_Multimodal_Initialization_API]]
* [[example_of::Vision_Language_Model_Patterns]]
