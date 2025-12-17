# File: `unsloth/kernels/moe/grouped_gemm/reference/moe_block.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 161 |
| Classes | `Qwen3MoeFusedGroupedGEMMBlock` |
| Imports | grouped_gemm, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Alternative reference implementation of Qwen3 MoE block with debugging features.

**Mechanism:** Qwen3MoeFusedGroupedGEMMBlock with additional intermediate result storage and checking utilities for detailed debugging.

**Significance:** Used for debugging and detailed intermediate result validation during kernel development.
