# File: `src/transformers/image_processing_utils_fast.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 953 |
| Classes | `BaseImageProcessorFast` |
| Functions | `validate_fast_preprocess_arguments`, `safe_squeeze`, `max_across_indices`, `get_max_height_width`, `divide_to_patches` |
| Imports | collections, copy, functools, huggingface_hub, image_processing_utils, image_transforms, image_utils, numpy, processing_utils, typing, ... +1 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides PyTorch/TorchVision-accelerated image processing with GPU support and optimized batch operations for fast preprocessing.

**Mechanism:** BaseImageProcessorFast leverages TorchVision's functional API (torch.nn.functional) to perform image operations on GPU tensors. It implements resize, crop, pad, rescale, and normalize operations using torch tensors instead of PIL/NumPy. Key optimizations include: batch processing via `group_images_by_shape()` to process same-sized images together, fused rescale+normalize operations to reduce memory ops, and automatic device placement. The class handles format conversions (PIL/NumPy → torch), validates preprocessing arguments, and provides extensive customization hooks through overrideable methods like `_preprocess()`, `_preprocess_image_like_inputs()`, and mask creation utilities.

**Significance:** This is the "fast path" for image preprocessing, offering 2-10x speedups over NumPy implementations when processing on GPU. It's especially critical for vision-language models and applications requiring real-time preprocessing. The design allows model-specific processors to inherit and customize behavior while maintaining performance optimizations.
