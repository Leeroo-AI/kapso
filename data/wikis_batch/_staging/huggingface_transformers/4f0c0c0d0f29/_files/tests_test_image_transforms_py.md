# File: `tests/test_image_transforms.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 647 |
| Classes | `ImageTransformsTester` |
| Functions | `get_random_image` |
| Imports | numpy, parameterized, transformers, unittest |

## Understanding

**Status:** ✅ Explored

**Purpose:** Unit tests for low-level image transformation functions used in image processing pipelines.

**Mechanism:** ImageTransformsTester validates core image operations including to_pil_image (with various dtypes and formats), to_channel_dimension_format, get_resize_output_image_size, resize, center_crop, pad, normalize, flip_channel_order, convert_to_rgb, and annotation format conversions (corners_to_center, center_to_corners, rgb_to_id, id_to_rgb). Tests cover numpy arrays, PIL images, and PyTorch tensors with both channels-first and channels-last layouts.

**Significance:** Ensures correctness of fundamental image preprocessing operations that underpin all image processors, validating proper handling of different data formats, channel orderings, and numeric precision requirements across the vision pipeline.
