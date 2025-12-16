# File: `unsloth/kernels/moe/grouped_gemm/reference/layers/llama4_moe.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 437 |
| Classes | `Llama4MoeResult`, `Llama4GroupedGemmTextMoe`, `Llama4TritonTextMoe` |
| Imports | dataclasses, grouped_gemm, torch, transformers, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides reference implementations of Llama 4 MoE blocks using grouped GEMM operations, both with torch-native and Triton-optimized kernels.

**Mechanism:** Implements three key components:
- `Llama4MoeResult`: Dataclass storing all intermediate results for debugging (token counts, routing weights, hidden states, expert outputs)
- `Llama4GroupedGemmTextMoe`: Extends HF's `Llama4TextMoe` by replacing expert computation with torch-native grouped GEMM, permuting weight matrices in-place and supporting optional overlap between router and shared expert computation
- `Llama4TritonTextMoe`: Further extends the grouped GEMM implementation by using Triton kernels instead of torch-native operations, with support for permutation fusion (permute_x/permute_y), autotuning, and manual kernel configuration

Key operations include router execution with sigmoid activation, token-to-expert permutation, two grouped GEMMs (gate_up_proj and down_proj) with SiLU activation between them, and combination with shared expert output.

**Significance:** Critical for testing and validating the Triton grouped GEMM kernels against known-good torch implementations. Provides the reference implementation for Llama 4's unique MoE architecture with shared experts and top-1 routing with sigmoid scoring.
