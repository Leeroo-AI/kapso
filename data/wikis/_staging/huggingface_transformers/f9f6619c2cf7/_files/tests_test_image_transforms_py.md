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

**Purpose:** Unit tests for low-level image transformation functions used in vision preprocessing pipelines.

**Mechanism:** The `ImageTransformsTester` class provides extensive parameterized tests for core image operations: PIL/numpy/torch tensor conversions (`to_pil_image`), channel dimension reordering (`to_channel_dimension_format`), resizing with various modes and aspect ratios (`resize`, `get_resize_output_image_size`), normalization with dtype preservation, center cropping with padding, bounding box format conversions (`center_to_corners_format`, `corners_to_center_format`), RGB/ID conversions for segmentation masks, padding with different modes (constant, reflect, replicate, symmetric), RGB conversion, and channel order flipping. Tests cover edge cases like ambiguous dimensions, float/int dtypes, and multi-channel images.

**Significance:** Ensures correctness of fundamental image processing operations that underpin all vision models. These transforms are building blocks used by image processors, so their reliability is critical. Tests validate numerical accuracy, dtype handling, and proper dimension ordering across different input formats.
