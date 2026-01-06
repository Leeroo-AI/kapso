# File: `examples/pooling/token_classify/ner.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 54 |
| Functions | `parse_args`, `main` |
| Imports | argparse, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Offline Named Entity Recognition example

**Mechanism:** Uses vLLM's `encode` method with `pooling_task="token_classify"` to perform NER on text. Loads NeuroBERT-NER model, gets per-token logits, maps token IDs to labels using model's id2label configuration, and prints entity annotations for each token (excluding special tokens).

**Significance:** Demonstrates token-level classification tasks in vLLM beyond sentence embeddings. Shows how to extract and interpret per-token predictions for sequence labeling tasks like NER, POS tagging, or semantic role labeling.
