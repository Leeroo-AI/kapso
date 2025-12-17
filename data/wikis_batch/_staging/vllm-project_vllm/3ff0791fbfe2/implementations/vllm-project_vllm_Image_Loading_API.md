= Image Loading API =

{{Metadata
| Knowledge Sources = vllm/multimodal/inputs.py, vllm/assets/image.py, examples/offline_inference/vision_language.py
| Domains = Image Processing, API Reference, Data Loading
| Last Updated = 2025-12-17
}}

== Overview ==

The '''Image Loading API''' implements the Multimodal Input Preparation Principle by providing a flexible interface for loading and preparing image inputs from various sources. This API supports multiple image formats and loading methods, ensuring compatibility with different use cases and workflows.

== Code Reference ==

=== Source Location ===

<syntaxhighlight lang="python">
# Type definitions
vllm/multimodal/inputs.py: ImageItem, ModalityData, MultiModalDataDict

# Asset utilities
vllm/assets/image.py: class ImageAsset

# Helper functions
vllm/multimodal/image.py: convert_image_mode()
</syntaxhighlight>

=== Type Definitions ===

<syntaxhighlight lang="python">
from typing import Union, List, Optional
from PIL.Image import Image
import numpy as np
import torch

# Single image item types
HfImageItem = Union[Image, np.ndarray, torch.Tensor]
ImageItem = Union[HfImageItem, torch.Tensor, MediaWithBytes[HfImageItem]]

# Modality data allows single item or list
ModalityData = Union[ImageItem, List[Optional[ImageItem]], None]

# Full multimodal data dictionary
MultiModalDataDict = Mapping[str, ModalityData]
</syntaxhighlight>

=== Import ===

<syntaxhighlight lang="python">
from PIL import Image
from vllm.assets.image import ImageAsset
from vllm.multimodal.image import convert_image_mode
</syntaxhighlight>

== Description ==

The Image Loading API provides several methods for loading images:

* '''PIL Image Loading''': Direct use of PIL.Image.open() for local files
* '''URL Loading''': Automatic downloading from HTTP/HTTPS URLs
* '''Base64 Encoding''': Support for base64-encoded image data
* '''Asset Utilities''': Built-in test images via ImageAsset class
* '''Format Conversion''': Helper functions to convert color modes (RGB, RGBA, etc.)

Images are loaded into PIL Image objects by default, which vLLM then processes using the model's HuggingFace processor.

== I/O Contract ==

=== Input Formats ===

{| class="wikitable"
! Input Type !! Description !! Example
|-
| PIL Image || Direct PIL Image object || Image.open("photo.jpg")
|-
| File Path || String path to local file || "path/to/image.png"
|-
| URL || HTTP/HTTPS URL string || "https://example.com/image.jpg"
|-
| Base64 || Base64-encoded string || "data:image/png;base64,iVBORw0..."
|-
| NumPy Array || NumPy ndarray (H, W, C) || np.array([...])
|-
| ImageAsset || Built-in test image || ImageAsset("cherry_blossom").pil_image
|}

=== Output ===

Returns a PIL Image object or the appropriate format for inclusion in `multi_modal_data` dictionary.

== Usage Examples ==

=== Loading from Local File ===

<syntaxhighlight lang="python">
from PIL import Image
from vllm import LLM, SamplingParams

# Load image from local file
image = Image.open("path/to/image.jpg")

llm = LLM(model="llava-hf/llava-1.5-7b-hf")

prompt = {
    "prompt": "USER: <image>\nWhat is in this image?\nASSISTANT:",
    "multi_modal_data": {"image": image}
}

outputs = llm.generate([prompt], SamplingParams(max_tokens=100))
print(outputs[0].outputs[0].text)
</syntaxhighlight>

=== Loading from URL ===

<syntaxhighlight lang="python">
from PIL import Image
import requests
from io import BytesIO

# Load image from URL
url = "https://example.com/photo.jpg"
response = requests.get(url)
image = Image.open(BytesIO(response.content))

prompt = {
    "prompt": "USER: <image>\nDescribe this image.\nASSISTANT:",
    "multi_modal_data": {"image": image}
}

outputs = llm.generate([prompt], SamplingParams(max_tokens=100))
</syntaxhighlight>

=== Using Built-in Test Assets ===

<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.multimodal.image import convert_image_mode

# Use built-in cherry blossom test image
image = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")

llm = LLM(model="llava-hf/llava-1.5-7b-hf")

prompt = {
    "prompt": "USER: <image>\nWhat is the content of this image?\nASSISTANT:",
    "multi_modal_data": {"image": image}
}

outputs = llm.generate([prompt], SamplingParams(max_tokens=100))
</syntaxhighlight>

=== Batch Image Loading ===

<syntaxhighlight lang="python">
from PIL import Image
from vllm import LLM, SamplingParams

# Load multiple images
image1 = Image.open("image1.jpg")
image2 = Image.open("image2.jpg")
image3 = Image.open("image3.jpg")

llm = LLM(
    model="llava-hf/llava-1.5-7b-hf",
    limit_mm_per_prompt={"image": 1}
)

prompts = [
    {
        "prompt": "USER: <image>\nDescribe this image.\nASSISTANT:",
        "multi_modal_data": {"image": image1}
    },
    {
        "prompt": "USER: <image>\nWhat do you see?\nASSISTANT:",
        "multi_modal_data": {"image": image2}
    },
    {
        "prompt": "USER: <image>\nAnalyze this image.\nASSISTANT:",
        "multi_modal_data": {"image": image3}
    }
]

outputs = llm.generate(prompts, SamplingParams(max_tokens=100))
for output in outputs:
    print(output.outputs[0].text)
    print("-" * 50)
</syntaxhighlight>

=== Multi-Image Support (for compatible models) ===

<syntaxhighlight lang="python">
from PIL import Image
from vllm import LLM, SamplingParams

# Some models support multiple images per prompt
image1 = Image.open("image1.jpg")
image2 = Image.open("image2.jpg")

llm = LLM(
    model="OpenGVLab/InternVL3-2B",
    limit_mm_per_prompt={"image": 2}  # Allow 2 images per prompt
)

# Pass list of images
prompt = {
    "prompt": "<image><image>\nCompare these two images.",
    "multi_modal_data": {"image": [image1, image2]}
}

outputs = llm.generate([prompt], SamplingParams(max_tokens=150))
</syntaxhighlight>

=== Loading with Format Conversion ===

<syntaxhighlight lang="python">
from PIL import Image
from vllm.multimodal.image import convert_image_mode

# Load and convert to RGB (removing alpha channel if present)
image = Image.open("image.png")
image_rgb = convert_image_mode(image, "RGB")

prompt = {
    "prompt": "USER: <image>\nWhat's in the image?\nASSISTANT:",
    "multi_modal_data": {"image": image_rgb}
}
</syntaxhighlight>

== Related Pages ==

* [[implements::Principle:vllm-project_vllm_Multimodal_Input_Preparation_Principle]]
* [[next_step::vllm-project_vllm_VLM_Prompt_Templates_Pattern]]
* [[uses::PIL_Image_Library]]
