# File: `src/transformers/pipelines/image_segmentation.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 223 |
| Classes | `ImageSegmentationPipeline` |
| Imports | base, numpy, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements pipeline for image segmentation tasks that detect and mask objects in images. Supports semantic, instance, and panoptic segmentation using various AutoModelForXXXSegmentation models.

**Mechanism:** The ImageSegmentationPipeline class extends Pipeline with three core methods: preprocess() loads images and prepares inputs using the image_processor, _forward() runs model inference, and postprocess() converts raw predictions to binary PIL Image masks with labels and scores. It handles multiple segmentation subtasks (semantic, instance, panoptic) by routing to appropriate post-processing methods based on model capabilities. Special handling is provided for OneFormer models which require tokenized task inputs.

**Significance:** Core component providing a unified interface for all image segmentation tasks in the transformers library. Essential for applications like scene understanding, object isolation, and visual analysis where identifying pixel-level regions is required. Returns masks as PIL images along with detected object labels and confidence scores.
