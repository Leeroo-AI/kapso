# File: `src/transformers/pipelines/zero_shot_image_classification.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 202 |
| Classes | `ZeroShotImageClassificationPipeline` |
| Imports | base, collections, typing, utils, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements zero-shot image classification using vision-language models (CLIP, SigLIP) to classify images against arbitrary text labels without task-specific training.

**Mechanism:** The `ZeroShotImageClassificationPipeline` loads images via `load_image`, processes them through the image processor, formats candidate labels with hypothesis template (default: "This is a photo of {}."), encodes text labels through tokenizer (with model-specific handling for SigLIP's max_length=64), computes vision-text similarity via `logits_per_image`, and applies softmax (CLIP) or sigmoid (SigLIP) normalization for probability scores. Returns ranked label predictions sorted by confidence.

**Significance:** Enables flexible image classification without pre-defined categories or model retraining. Crucial for scenarios with dynamic label sets, rare categories, or insufficient training data. Leverages contrastive vision-language pretraining to achieve strong zero-shot transfer across diverse visual concepts and supports multiple CLIP-family architectures.
