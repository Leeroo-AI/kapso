# File: `src/transformers/pipelines/image_to_image.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 145 |
| Classes | `ImageToImagePipeline` |
| Imports | base, numpy, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements a pipeline for image-to-image transformation tasks such as super-resolution, style transfer, and image enhancement using vision models.

**Mechanism:** The `ImageToImagePipeline` processes input images through an `AutoModelForImageToImage` model by preprocessing images with an image processor, running inference to obtain reconstruction outputs, and postprocessing the results by converting tensors back to PIL images with proper normalization and clamping. The pipeline handles tensor-to-image conversion by moving data to CPU, converting from CHW to HWC format, scaling from [0,1] float range to [0,255] uint8, and creating PIL Image objects.

**Significance:** This pipeline enables image transformation tasks where the output is another image of potentially different resolution or characteristics. It's commonly used for super-resolution models like Swin2SR that upscale images, as well as other image enhancement and transformation tasks that require maintaining image format while modifying content or quality.
