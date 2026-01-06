# File: `src/transformers/pipelines/image_feature_extraction.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 115 |
| Classes | `ImageFeatureExtractionPipeline` |
| Imports | base, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements a pipeline for extracting raw hidden state feature representations from vision models without any task-specific head.

**Mechanism:** The `ImageFeatureExtractionPipeline` extracts features by loading images, preprocessing them with an image processor, running them through a base vision transformer model, and returning either the raw hidden states or pooled outputs. Users can optionally request pooled embeddings via the `pool` parameter, and choose between tensor or list output formats via `return_tensors`. The pipeline accesses the model's internal representations (typically from the last hidden layer) rather than task-specific outputs.

**Significance:** This pipeline is essential for applications requiring vision embeddings as inputs to downstream tasks, such as similarity search, clustering, or custom classification layers. It provides a standardized way to extract feature vectors from pretrained vision models, enabling transfer learning and feature-based approaches without fine-tuning the entire model.
