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

**Purpose:** Core library of primitive image transformation functions for preprocessing, supporting NumPy arrays and PIL images with channel format handling.

**Mechanism:** Implements fundamental image operations as standalone functions: `resize()`, `rescale()`, `normalize()`, `center_crop()`, `pad()`, and channel manipulation (`to_channel_dimension_format()`, `flip_channel_order()`). All functions handle both channels-first (C,H,W) and channels-last (H,W,C) formats via the `input_data_format` parameter. Includes conversion utilities between PIL/NumPy/torch (`to_pil_image()`, automatic format detection via `infer_channel_dimension_format()`), bounding box format conversions (`center_to_corners_format()`, `corners_to_center_format()`), and utilities for grouped processing (`group_images_by_shape()`, `reorder_images()`) that enable efficient batch operations.

**Significance:** This is the fundamental transformation library used by both slow (NumPy) and fast (PyTorch) image processors. It provides format-agnostic, composable operations that maintain consistency across different backends. The functions mirror torchvision's design but with added flexibility for PIL integration and channel format handling required by diverse model architectures.
