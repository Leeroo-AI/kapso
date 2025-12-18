# File: `src/transformers/image_utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 959 |
| Classes | `ChannelDimension`, `AnnotationFormat`, `AnnotionFormat`, `ImageType`, `ImageFeatureExtractionMixin`, `SizeDict` |
| Functions | `is_pil_image`, `get_image_type`, `is_valid_image`, `is_valid_list_of_images`, `concatenate_list`, `valid_images`, `is_batched`, `is_scaled_image`, `... +17 more` |
| Imports | base64, collections, dataclasses, httpx, io, numpy, os, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Central utility module for image handling, validation, loading, and format detection across PIL, NumPy, and PyTorch representations.

**Mechanism:** Provides comprehensive image utilities organized into several categories: (1) Type detection and validation (`is_valid_image()`, `get_image_type()`, `is_pil_image()`, `is_batched()`), (2) Structure manipulation (`make_list_of_images()`, `make_flat_list_of_images()`, `make_nested_list_of_images()`), (3) Format inference and conversion (`infer_channel_dimension_format()`, `get_channel_dimension_axis()`, `get_image_size()`), (4) Image loading from various sources (`load_image()` supporting URLs, local paths, and base64), (5) Annotation validation for COCO detection/panoptic formats, and (6) ImageFeatureExtractionMixin providing legacy preprocessing methods. Defines enums for ChannelDimension, AnnotationFormat, and ImageType. Includes imagenet normalization constants.

**Significance:** This is the foundational utility layer that all image processing components depend on. It abstracts away the complexities of handling multiple image representations and provides a consistent interface for image validation, loading, and format handling across the library. Essential for robust preprocessing pipelines.
