# Implementation: OmniSearch_process_image

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::File|agent_eval.py|WebAgent/WebWatcher/infer/scripts_eval/agent_eval.py]]
|-
! Domains
| [[domain::Computer_Vision]], [[domain::Multimodal]], [[domain::Preprocessing]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==

Concrete image preprocessing method in the OmniSearch class that resizes images to meet pixel constraints and encodes them for API transport.

=== Description ===

The `process_image()` method in the `OmniSearch` class handles image preprocessing for the multimodal agent pipeline. It accepts images from various input formats, enforces pixel count constraints configured at initialization, converts to RGB format, and produces both a resized PIL Image object and a base64-encoded data URL suitable for inclusion in API requests.

The method uses the `smart_resize` utility from the Qwen-VL processor to calculate optimal dimensions that preserve aspect ratio while meeting the pixel constraints.

=== Usage ===

Use `process_image()` when:
- Preparing an image for the multimodal agent's main inference loop
- Converting user-uploaded images to API-ready format
- Ensuring images meet the model's resolution requirements

This method is called internally by `run_main()` at the start of each inference sample.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba-NLP/DeepResearch]
* '''File:''' WebAgent/WebWatcher/infer/scripts_eval/agent_eval.py
* '''Lines:''' 146-172

=== Signature ===
<syntaxhighlight lang="python">
def process_image(self, image: Union[str, Dict, PIL.Image]) -> Tuple[PIL.Image, str]:
    """
    Process and resize an image to meet pixel constraints.

    Args:
        image: Union[str, Dict, PIL.Image] - Input image as:
            - str: File path to image
            - Dict: Dictionary with "file_path" key
            - PIL.Image: Already loaded PIL Image object

    Returns:
        Tuple[PIL.Image, str]:
            - PIL.Image: Resized image in RGB format
            - str: Base64 data URL (data:image/jpeg;base64,...)

    Raises:
        FileNotFoundError: If image path does not exist
        PIL.UnidentifiedImageError: If file is not a valid image
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from scripts_eval.agent_eval import OmniSearch

# Initialize with pixel constraints
agent = OmniSearch(
    model_name="qwen-vl-plus",
    min_pixels=256*28*28,
    max_pixels=1280*28*28
)

# Process an image
resized_img, data_url = agent.process_image("/path/to/image.jpg")
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| image || Union[str, Dict, PIL.Image] || Yes || Image input in any supported format
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| resized_image || PIL.Image || Resized image meeting pixel constraints, RGB format
|-
| data_url || str || Base64-encoded data URL (data:image/jpeg;base64,...)
|}

== Usage Examples ==

=== Processing from File Path ===
<syntaxhighlight lang="python">
from scripts_eval.agent_eval import OmniSearch

agent = OmniSearch(model_name="qwen-vl-plus")

# Process image from file path
img, data_url = agent.process_image("/data/images/photo.jpg")
print(f"Resized to: {img.size}")  # e.g., (896, 672)
print(f"Data URL length: {len(data_url)}")  # Base64 string length
</syntaxhighlight>

=== Processing from Dictionary (Benchmark Format) ===
<syntaxhighlight lang="python">
# Common format from benchmark datasets
sample = {
    "file_path": "/data/hle/image_001.png",
    "prompt": "What is shown in this image?"
}

img, data_url = agent.process_image(sample)
# Or extract file_path first
img, data_url = agent.process_image(sample["file_path"])
</syntaxhighlight>

=== Processing PIL Image Directly ===
<syntaxhighlight lang="python">
import PIL.Image

# Load image with PIL first
pil_img = PIL.Image.open("/path/to/image.png")

# Process through the pipeline
resized, data_url = agent.process_image(pil_img)

# Image is now resized and base64 encoded
print(f"Original: {pil_img.size}, Resized: {resized.size}")
</syntaxhighlight>

=== Integration with Agent Loop ===
<syntaxhighlight lang="python">
from scripts_eval.agent_eval import OmniSearch

agent = OmniSearch(model_name="qwen-vl-plus")

sample = {
    "file_path": "/data/images/query.jpg",
    "prompt": "What species of bird is shown?"
}

# process_image is called internally by run_main
status, messages, answer = agent.run_main(sample)
print(f"Answer: {answer}")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Alibaba_NLP_DeepResearch_Image_Processing]]

=== Requires Environment ===
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_Python_Dependencies]]

=== Uses Heuristic ===
* [[uses_heuristic::Heuristic:Alibaba_NLP_DeepResearch_Image_Resizing_Constraints]]
