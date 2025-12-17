# File: `unsloth/models/qwen2.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 101 |
| Classes | `FastQwen2Model` |
| Imports | llama, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Qwen2 model adapter that reuses Llama optimizations with linear RoPE scaling.

**Mechanism:** Inherits from FastLlamaModel, patches Qwen2Attention forward with LlamaAttention_fast_forward, applies linear scaling RoPE module.

**Significance:** Minimal adapter demonstrating how similar architectures can reuse base optimizations with minimal code changes.
