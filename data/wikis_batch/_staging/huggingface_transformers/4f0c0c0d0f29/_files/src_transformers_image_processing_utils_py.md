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

**Purpose:** Base implementation for NumPy-based image processors with standard preprocessing operations and size handling utilities.

**Mechanism:** `BaseImageProcessor` extends `ImageProcessingMixin` with a `preprocess()` method that subclasses must implement. Provides three core operations delegating to image_transforms: `rescale()` (pixel scaling), `normalize()` (mean/std normalization), and `center_crop()` (with padding for undersized images). The `is_fast` property returns False (distinguishing from PyTorch-based fast processors). Size utilities handle the complexity of size specifications: `get_size_dict()` converts int/tuple/dict sizes to standardized dicts with valid key combinations (validated by `is_valid_size_dict()`): {"height", "width"} (exact), {"shortest_edge"} (aspect-preserving), {"shortest_edge", "longest_edge"} (constrained aspect), {"longest_edge"} (max dimension), or {"max_height", "max_width"} (independent bounds). `convert_to_size_dict()` handles conversion logic with `default_to_square` and `height_width_order` parameters. `select_best_resolution()` uses effective/wasted resolution heuristics for multi-resolution models. `get_patch_output_size()` calculates patch dimensions after resizing.

**Significance:** Standard base class for NumPy-based image processors (slower but more compatible), providing size standardization that eliminates ambiguity in resize operations while offering common preprocessing operations used by most vision models.
