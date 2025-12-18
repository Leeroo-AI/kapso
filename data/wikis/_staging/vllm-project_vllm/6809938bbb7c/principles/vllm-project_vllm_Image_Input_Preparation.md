{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|PIL Documentation|https://pillow.readthedocs.io]]
* [[source::Paper|Learning Transferable Visual Models|https://arxiv.org/abs/2103.00020]]
|-
! Domains
| [[domain::Vision]], [[domain::Image_Processing]], [[domain::Preprocessing]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

The process of loading, formatting, and preparing image data for consumption by vision-language models.

=== Description ===

Image Input Preparation transforms raw image data into model-ready format. This involves:

1. **Loading:** Reading images from files, URLs, or base64 data
2. **Format Conversion:** Ensuring correct color mode (RGB)
3. **Resizing:** Adjusting dimensions for model requirements
4. **Normalization:** Scaling pixel values appropriately
5. **Batching:** Organizing multiple images for efficient processing

The vision processor handles model-specific transformations automatically.

=== Usage ===

Prepare images when:
- Building visual question answering systems
- Creating image captioning pipelines
- Processing multimodal documents
- Running image-based chat interactions

== Theoretical Basis ==

'''Image Processing Pipeline:'''

<syntaxhighlight lang="text">
Raw Image → Load → RGB Convert → Resize → Normalize → Tensor
   (file)  (PIL)   (remove alpha) (336x336) ([-1,1])  (torch)
</syntaxhighlight>

'''Vision Encoder Preprocessing:'''

Most VLMs use CLIP-style preprocessing:

<syntaxhighlight lang="python">
# Standard CLIP preprocessing (conceptual)
def preprocess_image(image, target_size=336):
    # 1. Resize to square
    image = image.resize((target_size, target_size), Image.BICUBIC)

    # 2. Convert to float tensor
    tensor = torch.tensor(np.array(image)).float() / 255.0

    # 3. Normalize with CLIP stats
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
    tensor = (tensor - mean) / std

    # 4. Rearrange to CHW
    tensor = tensor.permute(2, 0, 1)

    return tensor
</syntaxhighlight>

'''Dynamic Resolution:'''

Some models support variable image sizes:

<syntaxhighlight lang="python">
# Dynamic patching (LLaVA-NeXT style)
def dynamic_preprocess(image, max_patches=6):
    w, h = image.size

    # Calculate optimal patch layout
    aspect_ratio = w / h
    best_grid = find_best_grid(aspect_ratio, max_patches)

    # Split into patches
    patches = split_into_grid(image, best_grid)

    # Also include thumbnail
    thumbnail = image.resize((336, 336))

    return [thumbnail] + patches
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_MultiModalData_image]]
