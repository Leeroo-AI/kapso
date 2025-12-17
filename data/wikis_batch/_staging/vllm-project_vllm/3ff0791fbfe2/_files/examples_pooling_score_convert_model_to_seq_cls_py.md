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

**Purpose:** Convert causal LM to classifier

**Mechanism:** Utility script that converts CausalLM reranker models into SequenceClassification format for vLLM compatibility. Provides two conversion methods: (1) `from_2_way_softmax` extracts yes/no token weights and computes their difference as the score weight (for 2-class models), and (2) `no_post_processing` directly copies classifier token weights. The converted model uses a linear classification head instead of the language modeling head.

**Significance:** Essential preprocessing tool enabling vLLM to efficiently serve reranker models. Bypasses full LM head computation by extracting only relevant scoring weights, improving inference speed for models like BGE-reranker-v2-gemma, mxbai-rerank, and Qwen3-Reranker.
