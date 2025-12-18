{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::Vision]], [[domain::NLP]], [[domain::Prompt_Engineering]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

Pattern documentation for constructing prompts that combine text with image placeholders for vision-language model inference.

=== Description ===

VLM prompts use special placeholder tokens (like `<image>`) to indicate where image embeddings should be inserted. Different models use different placeholder formats:
- LLaVA: `<image>` token in user content
- Qwen-VL: `<img>...</img>` tags
- Phi-3-Vision: `<|image_1|>` indexed placeholders

vLLM handles the model-specific templating automatically when using chat templates.

=== Usage ===

Construct VLM prompts when:
- Building image captioning prompts
- Creating visual QA queries
- Designing multi-image comparison prompts
- Implementing document understanding tasks

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project/vllm]
* '''File:''' examples/offline_inference/vision_language.py
* '''Lines:''' L42-200 (model-specific examples)

=== Pattern Specification ===
<syntaxhighlight lang="python">
# Standard multimodal prompt format
prompt_dict = {
    "prompt": str,           # Text with image placeholders
    "multi_modal_data": {
        "image": Image | list[Image] | str,  # Image data
    }
}

# Model-specific placeholder tokens:
# LLaVA:     "<image>"
# Qwen-VL:   "<img></img>" (URL between tags)
# Pixtral:   "[IMG]"
# Phi-3:     "<|image_1|>"
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| prompt || str || Yes || Text with model-appropriate image placeholder(s)
|-
| multi_modal_data || dict || Yes || Dict with "image" key containing image(s)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| prompt_dict || dict || Complete prompt ready for LLM.generate()
|}

== Usage Examples ==

=== LLaVA-Style Prompt ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from PIL import Image

llm = LLM(model="llava-hf/llava-1.5-7b-hf", trust_remote_code=True)

image = Image.open("photo.jpg")

# LLaVA uses <image> placeholder
prompt = "USER: <image>\nWhat is in this image?\nASSISTANT:"

outputs = llm.generate(
    [{
        "prompt": prompt,
        "multi_modal_data": {"image": image},
    }],
    SamplingParams(max_tokens=100),
)

print(outputs[0].outputs[0].text)
</syntaxhighlight>

=== Using Chat Template (Recommended) ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from PIL import Image

llm = LLM(model="llava-hf/llava-1.5-7b-hf", trust_remote_code=True)
tokenizer = llm.get_tokenizer()

image = Image.open("photo.jpg")

# Use chat template for automatic formatting
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},  # Image placeholder
            {"type": "text", "text": "What is in this image?"},
        ],
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

outputs = llm.generate(
    [{
        "prompt": prompt,
        "multi_modal_data": {"image": image},
    }],
    SamplingParams(max_tokens=100),
)
</syntaxhighlight>

=== Multi-Image Prompt ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from PIL import Image

llm = LLM(
    model="llava-hf/llava-v1.6-mistral-7b-hf",
    limit_mm_per_prompt={"image": 2},
    trust_remote_code=True,
)

images = [Image.open("img1.jpg"), Image.open("img2.jpg")]

# Multiple image placeholders
prompt = "USER: <image><image>\nCompare these two images.\nASSISTANT:"

outputs = llm.generate(
    [{
        "prompt": prompt,
        "multi_modal_data": {"image": images},
    }],
    SamplingParams(max_tokens=200),
)
</syntaxhighlight>

=== Phi-3 Vision Format ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from PIL import Image

llm = LLM(
    model="microsoft/Phi-3-vision-128k-instruct",
    trust_remote_code=True,
    mm_processor_kwargs={"max_dynamic_patch": 4},
)

image = Image.open("document.png")

# Phi-3 uses indexed placeholders
prompt = "<|user|>\n<|image_1|>\nExtract the text from this document.\n<|end|>\n<|assistant|>\n"

outputs = llm.generate(
    [{
        "prompt": prompt,
        "multi_modal_data": {"image": image},
    }],
    SamplingParams(max_tokens=500),
)
</syntaxhighlight>

=== Dynamic Prompt Construction ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from PIL import Image

def create_vlm_prompt(question, images, model_type="llava"):
    """Create VLM prompt based on model type."""

    if model_type == "llava":
        image_tokens = "<image>" * len(images)
        prompt = f"USER: {image_tokens}\n{question}\nASSISTANT:"
    elif model_type == "phi3":
        image_tokens = "".join(f"<|image_{i+1}|>" for i in range(len(images)))
        prompt = f"<|user|>\n{image_tokens}\n{question}\n<|end|>\n<|assistant|>\n"
    elif model_type == "pixtral":
        image_tokens = "[IMG]" * len(images)
        prompt = f"{image_tokens}\n{question}"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return {
        "prompt": prompt,
        "multi_modal_data": {"image": images if len(images) > 1 else images[0]},
    }

# Usage
images = [Image.open("photo.jpg")]
prompt_dict = create_vlm_prompt("What is this?", images, model_type="llava")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_VLM_Prompt_Construction]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
