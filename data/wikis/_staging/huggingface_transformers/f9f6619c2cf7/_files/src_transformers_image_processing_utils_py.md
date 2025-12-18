# File: `src/transformers/image_processing_utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 320 |
| Classes | `BaseImageProcessor` |
| Functions | `is_valid_size_dict`, `convert_to_size_dict`, `get_size_dict`, `select_best_resolution`, `get_patch_output_size` |
| Imports | collections, image_processing_base, image_transforms, image_utils, math, numpy, processing_utils, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements the base image processor class with standard preprocessing operations (resize, crop, normalize) and size dictionary utilities.

**Mechanism:** BaseImageProcessor extends ImageProcessingMixin to provide concrete implementations of common image transformations. It wraps functions from image_transforms module (rescale, normalize, center_crop) with consistent APIs. The file includes utilities for size parameter standardization: `get_size_dict()` converts various size formats (int, tuple, dict) into standardized dictionaries with keys like 'height', 'width', 'shortest_edge', or 'longest_edge'. Additional utilities include `select_best_resolution()` for choosing optimal resolution from candidates and `get_patch_output_size()` for computing output dimensions after cropping.

**Significance:** This serves as the foundation for NumPy-based image processors (the "slow" path), providing a reference implementation that model-specific processors can inherit and customize. It establishes the standard preprocessing API and ensures backwards compatibility through flexible size parameter handling.
