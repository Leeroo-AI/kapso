# File: `src/transformers/pipelines/image_segmentation.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 223 |
| Classes | `ImageSegmentationPipeline` |
| Imports | base, numpy, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements a unified pipeline for semantic, instance, and panoptic image segmentation that predicts masks and class labels for different regions or objects in images.

**Mechanism:** The `ImageSegmentationPipeline` supports multiple segmentation subtasks (semantic, instance, panoptic) through a flexible architecture. It preprocesses images while preserving target sizes, forwards them through segmentation models (supporting DETR, Mask2Former, OneFormer, and other architectures), and uses image processor post-processing methods to generate binary masks with associated labels and confidence scores. The pipeline automatically selects the appropriate post-processing method based on available capabilities and requested subtask, with special handling for OneFormer's task-based tokenization.

**Significance:** This is the primary user interface for all image segmentation tasks in transformers, consolidating multiple segmentation approaches (semantic, instance, panoptic) under a single unified API. It handles the complexity of different model architectures and output formats, providing users with consistent mask predictions regardless of the underlying model implementation.
