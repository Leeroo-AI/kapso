# File: `src/transformers/pipelines/image_feature_extraction.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 115 |
| Classes | `ImageFeatureExtractionPipeline` |
| Imports | base, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Extracts visual embeddings from images using vision transformer models without task-specific heads.

**Mechanism:** ImageFeatureExtractionPipeline loads images, preprocesses with image processor, runs through base vision model, returns either raw hidden states (model_outputs[0]) or pooled output (pooler_output) based on pool parameter, with option to return tensors or nested lists for downstream feature-based applications.

**Significance:** Vision equivalent of text feature extraction, enabling transfer learning for custom computer vision tasks like image similarity search, clustering, and retrieval by exposing raw image embeddings from pretrained ViT/BEiT/DeiT models without classification heads.
