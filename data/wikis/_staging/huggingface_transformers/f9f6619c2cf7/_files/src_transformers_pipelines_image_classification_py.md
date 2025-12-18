# File: `src/transformers/pipelines/image_classification.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 229 |
| Classes | `ClassificationFunction`, `ImageClassificationPipeline` |
| Functions | `sigmoid`, `softmax` |
| Imports | base, numpy, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements an image classification pipeline that predicts class labels and confidence scores for input images using vision transformer models.

**Mechanism:** The `ImageClassificationPipeline` class extends the base `Pipeline` to handle image classification tasks. It preprocesses images using an image processor, forwards them through an `AutoModelForImageClassification` model to obtain logits, and postprocesses the results by applying sigmoid or softmax activation functions (based on single-label vs multi-label classification) to produce scored class predictions. The pipeline supports batch processing, top-k filtering, and configurable activation functions through the `ClassificationFunction` enum.

**Significance:** This is a core user-facing pipeline that provides the standard interface for image classification tasks in transformers. It enables easy-to-use, high-level access to vision models for classifying images into predefined categories, handling all the complexity of image preprocessing, model inference, and output formatting behind a simple API.
