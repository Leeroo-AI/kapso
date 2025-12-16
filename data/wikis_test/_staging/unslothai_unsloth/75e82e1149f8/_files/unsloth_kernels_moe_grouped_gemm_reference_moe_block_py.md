# File: `unsloth/kernels/moe/grouped_gemm/reference/moe_block.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 161 |
| Classes | `Qwen3MoeFusedGroupedGEMMBlock` |
| Imports | grouped_gemm, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides a cleaner, production-focused implementation of Qwen3's fused grouped GEMM MoE block without debugging overhead.

**Mechanism:**
- Extends Qwen3MoeGroupedGEMMBlock (from layers/qwen3_moe.py)
- Uses fused Triton grouped_gemm kernels instead of torch_grouped_gemm
- Supports permute_x and permute_y fusions for efficient token routing
- Handles autotuning or manual kernel configuration
- Simplified forward pass without intermediate result storage
- Properly handles unpermutation based on permute_y flag

**Significance:** Production-ready MoE block for Qwen3 models. This is the recommended implementation for actual training/inference workloads, as it removes the debugging instrumentation present in the reference layers. Balances performance with maintainability by building on the validated reference implementation.