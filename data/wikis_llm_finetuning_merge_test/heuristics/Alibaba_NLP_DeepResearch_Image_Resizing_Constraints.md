# Heuristic: Image_Resizing_Constraints

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::Experience|Internal|Code analysis of agent_eval.py]]
|-
! Domains
| [[domain::Multimodal]], [[domain::Image_Processing]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==
Image preprocessing constraints using max_pixels (1024*28*28) and min_pixels (256*28*28) to optimize vision-language model performance.

=== Description ===
The multimodal OmniSearch agent processes images before sending them to the vision-language model. Images are resized to fall within a specific pixel range: not too large to cause memory issues, not too small to lose important details. The constraints use multipliers of 28 (the typical patch size for vision transformers) to ensure clean divisibility.

=== Usage ===
Use this heuristic when preprocessing images for the multimodal agent. Apply these constraints to ensure consistent input quality and prevent VRAM overflow during inference with large images.

== The Insight (Rule of Thumb) ==
* **Action:** Resize images to fit within the pixel constraints before VLM inference.
* **Max Pixels:** `1024 * 28 * 28` = 802,816 pixels (~896x896 or equivalent area)
* **Min Pixels:** `256 * 28 * 28` = 200,704 pixels (~448x448 or equivalent area)
* **Trade-off:** Upscaling small images may introduce artifacts; downscaling large images loses detail.
* **Aspect Ratio:** Preserved during resize using proportional scaling factor.

== Reasoning ==
Vision-language models have optimal input resolution ranges:

1. **Memory efficiency:** Large images consume significant VRAM during attention computation
2. **Patch alignment:** Resolutions that are multiples of patch size (28) process more efficiently
3. **Detail preservation:** Minimum size ensures enough pixels for meaningful visual features
4. **Batch consistency:** Consistent size ranges enable efficient batched inference

The formula `sqrt(target_pixels / (width * height))` maintains aspect ratio while hitting the target pixel count.

== Code Evidence ==

Image preprocessing from `agent_eval.py:138-160`:
<syntaxhighlight lang="python">
self.max_pixels = 1024 * 28 * 28
self.min_pixels = 256 * 28 * 28

def process_image(self, image):
    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))
    elif isinstance(image, str):
        image = Image.open(image)

    if (image.width * image.height) > self.max_pixels:
        resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < self.min_pixels:
        resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))
</syntaxhighlight>

Alternative constraints from `generation.py:21`:
<syntaxhighlight lang="python">
def process_image(image, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
</syntaxhighlight>

General image resize utility from `utils.py:519-533`:
<syntaxhighlight lang="python">
def resize_image(img, short_side_length: int = 1080):
    """Resize image by constraining short side to specified length."""
    # ... aspect ratio preserving resize
    resized_img = img.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)
    return resized_img
</syntaxhighlight>

== Related Pages ==
* [[used_by::Implementation:Alibaba_NLP_DeepResearch_OmniSearch_process_image]]
* [[used_by::Implementation:Alibaba_NLP_DeepResearch_OmniSearch_run_main]]
* [[used_by::Principle:Alibaba_NLP_DeepResearch_Image_Processing]]
