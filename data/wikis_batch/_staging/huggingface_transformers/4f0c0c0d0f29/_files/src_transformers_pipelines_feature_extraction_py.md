# File: `src/transformers/pipelines/feature_extraction.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 88 |
| Classes | `FeatureExtractionPipeline` |
| Imports | base, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Extracts hidden state embeddings from transformer models without task-specific heads for use as features in downstream applications.

**Mechanism:** FeatureExtractionPipeline tokenizes input text, runs through base transformer model to get hidden states (model_outputs[0] which is last_hidden_state), and returns either raw tensors or nested lists of floats representing the contextual embeddings for each token.

**Significance:** Fundamental building block for transfer learning and semantic applications like sentence similarity, clustering, and retrieval where raw embeddings are needed rather than task-specific predictions, enabling users to build custom downstream tasks on top of pretrained representations.
