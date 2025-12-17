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

**Purpose:** High-performance PyTorch/TorchVision-based image processor with GPU acceleration, batch optimization, and a complete implementation requiring minimal subclass code.

**Mechanism:** `BaseImageProcessorFast` extends `BaseImageProcessor` with `is_fast=True`. The preprocessing pipeline: (1) `preprocess()` validates kwargs using `validate_typed_dict()`, sets defaults from instance attributes, and delegates to `_preprocess_image_like_inputs()`, (2) `_prepare_image_like_inputs()` converts PIL/numpy/torch images to channel-first torch tensors via `_process_image()` (handling RGB conversion, format normalization, device placement), (3) `_preprocess()` groups images by shape using `group_images_by_shape()` for efficient batched operations, applies transformations (resize via torchvision's F.resize with interpolation mapping, center_crop with padding support, fused rescale+normalize via `_fuse_mean_std_and_rescale_factor()` reducing operations, optional padding to uniform size), and reorders via `reorder_images()`. The grouping/reordering pattern minimizes GPU kernel launches. Includes utilities like `divide_to_patches()` for patch-based models and `compile_friendly_resize()` for torch.compile compatibility. The architecture allows customization via overriding `_preprocess()` (most common), `_preprocess_image_like_inputs()` (multi-input), `_further_process_kwargs()` (custom args), or `_validate_preprocess_kwargs()` (validation).

**Significance:** Production-ready fast image processor reducing preprocessing bottlenecks for vision models, especially important for real-time inference and large-batch training. The extensive base implementation means most processors only need to set class attributes rather than implement processing logic.
