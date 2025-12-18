{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|PIL Documentation|https://pillow.readthedocs.io]]
|-
! Domains
| [[domain::Vision]], [[domain::Image_Processing]], [[domain::Multimodal]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

Pattern documentation for preparing image inputs in various formats (PIL, URLs, base64) for vision-language model inference.

=== Description ===

vLLM accepts images in multiple formats:
- **PIL.Image:** In-memory image objects
- **URL strings:** Remote images fetched automatically
- **Base64 strings:** Encoded image data
- **File paths:** Local image files (with appropriate configuration)

Images are preprocessed by model-specific vision processors before being passed to the model.

=== Usage ===

Prepare image inputs when:
- Building image captioning pipelines
- Creating visual question answering systems
- Processing documents with embedded images
- Running multimodal chat applications

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project/vllm]
* '''File:''' vllm/multimodal/image.py
* '''Lines:''' L1-100

=== Pattern Specification ===
<syntaxhighlight lang="python">
# Supported image input formats
from PIL import Image

# Format 1: PIL Image object
image = Image.open("photo.jpg")

# Format 2: URL string
image = "https://example.com/image.jpg"

# Format 3: Base64 encoded
import base64
with open("photo.jpg", "rb") as f:
    image = f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"

# Format 4: Multiple images (list)
images = [Image.open("img1.jpg"), Image.open("img2.jpg")]
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from PIL import Image
import requests
from io import BytesIO
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| image || PIL.Image || Conditional || Loaded PIL Image object
|-
| url || str || Conditional || URL to remote image
|-
| base64 || str || Conditional || Base64-encoded image with data URI prefix
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| multi_modal_data || dict || Dict with "image" key containing processed image(s)
|}

== Usage Examples ==

=== PIL Image from File ===
<syntaxhighlight lang="python">
from PIL import Image

# Load local image
image = Image.open("path/to/photo.jpg")

# Use in prompt
prompt_dict = {
    "prompt": "Describe this image: <image>",
    "multi_modal_data": {"image": image},
}
</syntaxhighlight>

=== Image from URL ===
<syntaxhighlight lang="python">
import requests
from PIL import Image
from io import BytesIO

# Download image
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png"
response = requests.get(url)
image = Image.open(BytesIO(response.content))

# Or pass URL directly (vLLM fetches it)
prompt_dict = {
    "prompt": "What is in this image? <image>",
    "multi_modal_data": {"image": url},
}
</syntaxhighlight>

=== Multiple Images ===
<syntaxhighlight lang="python">
from PIL import Image

# Load multiple images
images = [
    Image.open("image1.jpg"),
    Image.open("image2.jpg"),
]

# Prompt with multiple image placeholders
prompt_dict = {
    "prompt": "Compare these images: <image> and <image>",
    "multi_modal_data": {"image": images},
}
</syntaxhighlight>

=== Base64 Encoded Image ===
<syntaxhighlight lang="python">
import base64

# Encode image as base64
with open("photo.jpg", "rb") as f:
    base64_data = base64.b64encode(f.read()).decode()

# Use data URI format
image_uri = f"data:image/jpeg;base64,{base64_data}"

prompt_dict = {
    "prompt": "Describe this: <image>",
    "multi_modal_data": {"image": image_uri},
}
</syntaxhighlight>

=== Image Preprocessing ===
<syntaxhighlight lang="python">
from PIL import Image

# Preprocess image before submission
image = Image.open("large_photo.jpg")

# Resize if needed (some models have size limits)
max_size = 1024
if max(image.size) > max_size:
    image.thumbnail((max_size, max_size))

# Convert to RGB (remove alpha channel)
if image.mode != "RGB":
    image = image.convert("RGB")

prompt_dict = {
    "prompt": "Analyze this image: <image>",
    "multi_modal_data": {"image": image},
}
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_Image_Input_Preparation]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
