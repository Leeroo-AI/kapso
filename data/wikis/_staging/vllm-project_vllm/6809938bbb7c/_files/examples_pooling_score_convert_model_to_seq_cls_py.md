# File: `examples/pooling/score/convert_model_to_seq_cls.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 134 |
| Functions | `from_2_way_softmax`, `no_post_processing`, `converting`, `parse_args` |
| Imports | argparse, json, torch, transformers |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Model conversion utility for reranker models

**Mechanism:** Converts CausalLM models to SequenceClassification models for reranking by extracting classifier weights from the language model head. Supports two methods: (1) `from_2_way_softmax` - extracts weights for binary classification tokens and computes their difference, (2) `no_post_processing` - directly uses weights from classifier tokens. Handles models like bge-reranker-v2-gemma, mxbai-rerank, and Qwen3-Reranker. Saves converted models with appropriate configurations.

**Significance:** Critical utility for deploying reranker models in vLLM. Many reranking models are trained as CausalLMs with special output tokens but need to be converted to SequenceClassification format for efficient inference. Enables vLLM to serve these models with proper scoring semantics rather than next-token prediction.
