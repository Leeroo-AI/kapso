# Principle: Image_Processing

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

Image preprocessing for vision-language models. Includes resizing to meet pixel constraints, format conversion, and base64 encoding for API transport.

=== Description ===

Image Processing in the context of vision-language models involves preparing visual inputs for consumption by multimodal APIs. The preprocessing pipeline ensures images meet the model's constraints while preserving visual quality and semantic content.

The key processing steps include:

1. **Image Loading** - Accept various input formats (file paths, PIL images, URLs, dictionaries)
2. **Pixel Constraint Enforcement** - Resize images to meet minimum and maximum pixel count requirements
3. **Format Conversion** - Convert to RGB color space for consistent processing
4. **Base64 Encoding** - Encode images as data URLs for HTTP API transport

The pixel constraints (min_pixels, max_pixels) prevent:
- Images too small to provide meaningful visual information
- Images too large that would exceed memory or API limits

=== Usage ===

Use image processing when:
- Preparing images for vision-language model APIs (Qwen-VL, GPT-4V, etc.)
- Converting user-uploaded images for multimodal agents
- Ensuring consistent image format across diverse input sources

Image processing is performed once per image at the start of inference, before the image enters the agent loop.

== Theoretical Basis ==

The preprocessing pipeline:

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# Image preprocessing pipeline
def process_image(image, min_pixels, max_pixels) -> (PIL.Image, str):
    # Step 1: Load image from various sources
    if isinstance(image, str):
        img = PIL.Image.open(image)
    elif isinstance(image, dict):
        img = PIL.Image.open(image["file_path"])
    else:
        img = image  # Already PIL.Image

    # Step 2: Convert to RGB
    img = img.convert("RGB")

    # Step 3: Calculate resize factor based on pixel constraints
    width, height = img.size
    current_pixels = width * height

    if current_pixels > max_pixels:
        scale = math.sqrt(max_pixels / current_pixels)
        new_size = (int(width * scale), int(height * scale))
        img = img.resize(new_size, resample=PIL.Image.LANCZOS)
    elif current_pixels < min_pixels:
        scale = math.sqrt(min_pixels / current_pixels)
        new_size = (int(width * scale), int(height * scale))
        img = img.resize(new_size, resample=PIL.Image.LANCZOS)

    # Step 4: Base64 encode for API transport
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    base64_data = base64.b64encode(buffer.getvalue()).decode()
    data_url = f"data:image/jpeg;base64,{base64_data}"

    return img, data_url
</syntaxhighlight>

Typical pixel constraints:
- '''min_pixels:''' 256 * 28 * 28 (ensures sufficient detail)
- '''max_pixels:''' 1280 * 28 * 28 (prevents memory overflow)

The LANCZOS resampling filter provides high-quality results for both upscaling and downscaling operations.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_OmniSearch_process_image]]
