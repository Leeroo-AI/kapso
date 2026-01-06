# File: `src/transformers/pipelines/object_detection.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 197 |
| Classes | `ObjectDetectionPipeline` |
| Imports | base, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements a pipeline for detecting objects in images by predicting bounding boxes, class labels, and confidence scores using object detection models.

**Mechanism:** The `ObjectDetectionPipeline` supports two types of models: standard object detection models (like DETR) and LayoutLM variants for document understanding. For standard models, it preprocesses images while tracking target sizes, runs inference to obtain bounding box predictions and class logits, and uses the image processor's post-processing to apply NMS and threshold filtering. For LayoutLM models, it additionally tokenizes OCR-extracted text and boxes, classifies words, and unnormalizes coordinates. The `_get_bounding_box()` helper converts tensor coordinates to dictionary format with xmin, ymin, xmax, ymax keys.

**Significance:** This is the core pipeline for object detection tasks in transformers, providing a unified interface for detecting and localizing objects in natural images and documents. It's essential for applications like autonomous driving, surveillance, visual search, and document analysis, handling the complexity of different model architectures while providing consistent bounding box predictions with class labels and confidence scores.
