# File: `src/transformers/pipelines/feature_extraction.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 88 |
| Classes | `FeatureExtractionPipeline` |
| Imports | base, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Extracts raw hidden state embeddings from transformer models without any task-specific head, providing dense vector representations of input text.

**Mechanism:** The FeatureExtractionPipeline is one of the simplest pipelines - it tokenizes input text, runs it through the transformer model, and returns the last hidden states (or logits if available) either as tensors or nested lists. Unlike task-specific pipelines, it applies no post-processing like classification or decoding, instead exposing the raw contextual embeddings from the model's final layer. These embeddings capture semantic meaning and can be used for downstream tasks like similarity search, clustering, or as input features to other models.

**Significance:** Serves as the foundation for embedding-based applications like semantic search, document clustering, and recommendation systems. By providing direct access to transformer representations without task-specific constraints, it enables developers to use pre-trained language models as general-purpose feature extractors for custom machine learning workflows that go beyond the standard NLP tasks.
