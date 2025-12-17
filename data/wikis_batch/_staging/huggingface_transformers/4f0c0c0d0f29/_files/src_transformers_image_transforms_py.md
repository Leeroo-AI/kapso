# File: `src/transformers/image_transforms.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 1001 |
| Classes | `PaddingMode` |
| Functions | `to_channel_dimension_format`, `rescale`, `to_pil_image`, `get_size_with_aspect_ratio`, `get_resize_output_image_size`, `resize`, `normalize`, `center_crop`, `... +10 more` |
| Imports | collections, image_utils, math, numpy, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive library of image transformation primitives for preprocessing, supporting both NumPy arrays and PyTorch tensors with channel dimension flexibility.

**Mechanism:** Core transformations: `rescale()` (pixel value scaling with dtype control), `resize()` (PIL-based with resampling filters and reducing_gap optimization), `normalize()` (channel-wise mean/std with automatic float casting), `center_crop()` (with zero-padding for undersized images), `pad()` (supporting constant/reflect/replicate/symmetric modes via PaddingMode enum), `to_channel_dimension_format()` (FIRST↔LAST transposition preserving leading batch dims), and `convert_to_rgb()` (PIL.Image mode conversion). Utility functions: `get_resize_output_image_size()` (computes target size from int/tuple/max_size with aspect ratio preservation), `get_size_with_aspect_ratio()` (aspect-constrained resizing), `to_pil_image()` (handles numpy/torch→PIL with auto-rescaling detection via `_rescale_for_pil_conversion()`), and `flip_channel_order()` (RGB↔BGR). Bounding box utilities: `center_to_corners_format()` and `corners_to_center_format()` with both NumPy and PyTorch implementations. Panoptic segmentation helpers: `rgb_to_id()` and `id_to_rgb()` for color↔ID conversion. Batch utilities: `group_images_by_shape()` (groups by shape for efficient batch processing with optional paired inputs), `reorder_images()` (reconstructs original order), `split_to_tiles()` (divides images into grid tiles).

**Significance:** Low-level transformation library used by all image processors (both slow and fast variants), providing consistent image manipulation primitives with careful handling of channel dimensions, data types, and aspect ratios. The batch grouping utilities enable efficient GPU processing in fast processors.
