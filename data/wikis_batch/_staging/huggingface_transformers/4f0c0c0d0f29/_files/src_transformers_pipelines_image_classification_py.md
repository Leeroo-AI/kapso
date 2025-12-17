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

**Purpose:** Classifies images into predefined categories using AutoModelForImageClassification models.

**Mechanism:** ImageClassificationPipeline loads images via load_image() supporting URLs/paths/PIL, preprocesses with image processor, runs through vision model, applies sigmoid (single-label) or softmax (multi-label) activation functions based on model config or user specification, sorts predictions by confidence, and returns top-k label-score pairs.

**Significance:** Primary interface for image categorization tasks enabling applications like content moderation, medical image diagnosis, and visual search by providing easy access to vision models (ViT, BEiT, DeiT, Swin) with automatic score calibration and flexible function_to_apply control.
