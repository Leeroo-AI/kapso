# File: `src/transformers/pipelines/zero_shot_object_detection.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 242 |
| Classes | `ZeroShotObjectDetectionPipeline` |
| Imports | base, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements zero-shot object detection using vision-language models (OWL-ViT) to detect and localize objects in images based on text descriptions without task-specific training.

**Mechanism:** The `ZeroShotObjectDetectionPipeline` (extends ChunkPipeline) processes each candidate label separately by loading images, tokenizing label text, encoding through image and text encoders, computing vision-text alignment to generate bounding box predictions with confidence scores, filtering by threshold (default 0.1), and optionally limiting to top-k detections. Supports flexible input formats including single images, batches, and datasets. Returns bounding boxes (xmin, ymin, xmax, ymax) with associated labels and scores in original image coordinates.

**Significance:** Enables open-vocabulary object detection without predefined object categories or detection-specific training. Revolutionary for applications requiring flexible object queries, rare object detection, or dynamic category definitions. Extends zero-shot paradigm from classification to spatial localization, making it possible to detect arbitrary objects described in natural language.
