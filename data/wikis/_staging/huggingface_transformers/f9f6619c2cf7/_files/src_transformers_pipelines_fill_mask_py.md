# File: `src/transformers/pipelines/fill_mask.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 259 |
| Classes | `FillMaskPipeline` |
| Imports | base, numpy, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Predicts masked tokens in text using masked language models like BERT, enabling fill-in-the-blank style inference.

**Mechanism:** The FillMaskPipeline tokenizes input text containing [MASK] tokens (or model-specific mask tokens), identifies masked positions using get_masked_index(), runs the model to get logits at mask positions, applies softmax to get probabilities, and returns top-k predictions. It supports optional target filtering to restrict predictions to specific vocabulary items via get_target_ids(). For each prediction, it reconstructs the full sequence with the mask replaced and returns the completed text along with token IDs, token strings, and confidence scores. The pipeline handles both single-mask and experimental multi-mask scenarios.

**Significance:** Demonstrates the core capability of masked language models and enables practical applications like text completion, spell correction, and testing model understanding of context. While simpler than generative tasks, fill-mask showcases bidirectional understanding and is used for probing model knowledge, data augmentation, and interactive text editing tools.
