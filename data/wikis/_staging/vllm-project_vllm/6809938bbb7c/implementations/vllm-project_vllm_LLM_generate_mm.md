{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::Vision]], [[domain::NLP]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

Concrete tool for executing multimodal generation with combined text and image inputs on vision-language models.

=== Description ===

`LLM.generate()` with multimodal prompts processes combined vision-language inputs. The generate method:
- Accepts prompt dicts with `multi_modal_data` field
- Preprocesses images through the vision encoder
- Combines image embeddings with text tokens
- Generates text responses describing or analyzing images

This is the same `generate()` method used for text-only inference, extended for multimodal inputs.

=== Usage ===

Use multimodal generation when:
- Running image captioning
- Performing visual question answering
- Analyzing documents with images
- Creating image-based chat responses

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project/vllm]
* '''File:''' vllm/entrypoints/llm.py
* '''Lines:''' L365-434

=== Signature ===
<syntaxhighlight lang="python">
def generate(
    self,
    prompts: PromptType | Sequence[PromptType],
    # Multimodal prompts are dicts with:
    # - "prompt": str with image placeholders
    # - "multi_modal_data": {"image": Image | list[Image]}
    sampling_params: SamplingParams | None = None,
    **kwargs,
) -> list[RequestOutput]:
    """Generate with optional multimodal inputs."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from PIL import Image
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| prompts || list[dict] || Yes || Prompt dicts with "prompt" and "multi_modal_data" keys
|-
| sampling_params || SamplingParams || No || Generation parameters
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| outputs || list[RequestOutput] || Generated text descriptions/answers
|}

== Usage Examples ==

=== Basic Image Captioning ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from PIL import Image

llm = LLM(model="llava-hf/llava-1.5-7b-hf", trust_remote_code=True)

image = Image.open("photo.jpg")

outputs = llm.generate(
    [{
        "prompt": "USER: <image>\nDescribe this image in detail.\nASSISTANT:",
        "multi_modal_data": {"image": image},
    }],
    SamplingParams(temperature=0.2, max_tokens=200),
)

print(outputs[0].outputs[0].text)
</syntaxhighlight>

=== Batch Image Processing ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from PIL import Image
from pathlib import Path

llm = LLM(model="llava-hf/llava-1.5-7b-hf", trust_remote_code=True)

# Load multiple images
image_dir = Path("images/")
image_files = list(image_dir.glob("*.jpg"))

prompts = []
for img_path in image_files:
    image = Image.open(img_path)
    prompts.append({
        "prompt": "USER: <image>\nCaption this image.\nASSISTANT:",
        "multi_modal_data": {"image": image},
    })

# Process batch
outputs = llm.generate(prompts, SamplingParams(max_tokens=100))

for img_path, output in zip(image_files, outputs):
    print(f"{img_path.name}: {output.outputs[0].text}")
</syntaxhighlight>

=== Visual Question Answering ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from PIL import Image

llm = LLM(model="llava-hf/llava-1.5-7b-hf", trust_remote_code=True)

image = Image.open("chart.png")

questions = [
    "What type of chart is this?",
    "What is the highest value shown?",
    "What trend does this data show?",
]

prompts = [
    {
        "prompt": f"USER: <image>\n{q}\nASSISTANT:",
        "multi_modal_data": {"image": image},
    }
    for q in questions
]

outputs = llm.generate(prompts, SamplingParams(max_tokens=100))

for q, output in zip(questions, outputs):
    print(f"Q: {q}")
    print(f"A: {output.outputs[0].text}\n")
</syntaxhighlight>

=== Multi-Image Comparison ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from PIL import Image

llm = LLM(
    model="llava-hf/llava-v1.6-mistral-7b-hf",
    limit_mm_per_prompt={"image": 2},
    trust_remote_code=True,
)

img1 = Image.open("before.jpg")
img2 = Image.open("after.jpg")

outputs = llm.generate(
    [{
        "prompt": "USER: <image><image>\nCompare these two images. What changed?\nASSISTANT:",
        "multi_modal_data": {"image": [img1, img2]},
    }],
    SamplingParams(max_tokens=300),
)

print(outputs[0].outputs[0].text)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_Multimodal_Generation]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
