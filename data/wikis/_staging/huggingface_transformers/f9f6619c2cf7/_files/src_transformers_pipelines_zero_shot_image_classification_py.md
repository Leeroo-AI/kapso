# File: `src/transformers/pipelines/zero_shot_image_classification.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 202 |
| Classes | `ZeroShotImageClassificationPipeline` |
| Imports | base, collections, typing, utils, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements zero-shot image classification pipeline that classifies images without task-specific training using vision-language models like CLIP.

**Mechanism:** The `ZeroShotImageClassificationPipeline` class extends the base Pipeline and works by processing images through an image processor and tokenizing candidate labels formatted with a hypothesis template (default: "This is a photo of {}"). It passes both image and text inputs through the model (e.g., CLIPModel) to compute similarity scores via `logits_per_image`. For most models it applies softmax to get probabilities, but uses sigmoid for SigLIP models. Results are sorted by score and returned as label-score dictionaries.

**Significance:** Critical user-facing pipeline that enables powerful zero-shot classification capabilities without retraining. Allows users to classify images into arbitrary categories by simply providing text labels, making it highly flexible for diverse classification tasks across domains.
