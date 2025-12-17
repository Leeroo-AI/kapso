# File: `unsloth/models/qwen3_moe.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 243 |
| Classes | `FastQwen3MoeModel` |
| Functions | `Qwen3MoeSparseMoeBlock_fast_forward`, `Qwen3MoeDecoderLayer_fast_forward` |
| Imports | _utils, llama, os, qwen3, transformers, unsloth_zoo |

## Understanding

**Status:** ✅ Explored

**Purpose:** Qwen3 MoE (Mixture of Experts) sparse attention and routing optimization.

**Mechanism:** Patches Qwen3MoeSparseMoeBlock with efficient sparse forward, reuses Qwen3Attention optimizations, manages expert routing and load balancing.

**Significance:** Extends framework to sparse MoE models with routing-aware optimization for efficient expert selection.
