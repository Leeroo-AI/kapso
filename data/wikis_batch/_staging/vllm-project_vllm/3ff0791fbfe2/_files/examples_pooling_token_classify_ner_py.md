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

**Purpose:** Named entity recognition offline

**Mechanism:** This script performs offline NER using vLLM's token classification capability. It loads a token classification model (NeuroBERT-NER), runs `llm.encode()` with `pooling_task="token_classify"` to get per-token logits, applies argmax to get label predictions, and maps them to entity types using the model's label configuration. The output shows each token with its predicted entity label.

**Significance:** Example demonstrating vLLM's support for token-level classification tasks beyond embeddings and generation. Shows how to extract structured information (named entities) from text using specialized models.
