# File: `tests/test_image_processing_common.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 805 |
| Classes | `ImageProcessingTestMixin`, `AnnotationFormatTestMixin` |
| Functions | `prepare_image_inputs`, `prepare_video`, `prepare_video_inputs` |
| Imports | copy, datetime, httpx, inspect, io, json, numpy, os, packaging, pathlib, ... +7 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive testing framework for image and video processors, including both slow (PIL-based) and fast (torchvision-based) implementations.

**Mechanism:** Provides `ImageProcessingTestMixin` with extensive tests covering: slow/fast equivalence validation, serialization/deserialization, AutoImageProcessor compatibility, cross-loading (fast from slow, slow from fast), dtype/device casting, different input formats (PIL, numpy, torch tensors), multi-channel images, compilation support, and fast processor requirements for new models. Helper functions `prepare_image_inputs`, `prepare_video`, and `prepare_video_inputs` generate test data. Also includes `AnnotationFormatTestMixin` for validating legacy annotation format support in object detection/segmentation tasks.

**Significance:** Critical quality assurance for vision preprocessing pipeline. Ensures image processors maintain numerical accuracy across implementations, can be compiled for performance, properly handle various input formats, and maintain backward compatibility. The fast/slow equivalence tests are particularly important for validating optimized implementations against reference implementations.
