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

**Purpose:** Optimized implementation for Qwen3 Mixture-of-Experts models

**Mechanism:** Extends `FastQwen3Model` with MoE-specific optimizations:
- Sparse expert routing and load balancing
- Efficient expert parallel execution
- Token routing optimization to minimize communication
- Top-K expert selection with optimized dispatch

**Significance:** Supports Qwen3-30B-A3B and similar sparse MoE variants. MoE models activate only a subset of experts per token, requiring specialized routing logic for efficiency.
