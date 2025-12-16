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

**Purpose:** Qwen3 mixture-of-experts variant

**Mechanism:** Extends Qwen3 with sparse mixture-of-experts routing. Implements expert selection, gating, and routing weight computation with efficient expert parallelism using index operations.

**Significance:** Demonstrates MoE pattern integration with standard transformer layers. Shows how to combine attention optimization with expert routing for sparse models.
