# File: `unsloth/kernels/moe/grouped_gemm/reference/layers/qwen3_moe.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 348 |
| Classes | `GroupedGEMMResult`, `Qwen3MoeGroupedGEMMBlock`, `Qwen3MoeFusedGroupedGEMMBlock` |
| Imports | dataclasses, grouped_gemm, torch, transformers, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements reference MoE blocks for Qwen3 models using grouped GEMM operations, providing both torch-native and Triton-accelerated versions for testing and validation.

**Mechanism:** Provides three main components:
- `GroupedGEMMResult`: Dataclass capturing intermediate results (token counts, routing indices, topk weights, first/second GEMM outputs, unpermuted states, final output)
- `Qwen3MoeGroupedGEMMBlock`: Torch-native grouped GEMM implementation that extracts weights from HuggingFace's `Qwen3MoeSparseMoeBlock`, performs router computation with softmax + topk + normalization, token permutation, two grouped GEMMs with activation, unpermutation, and weight merging
- `Qwen3MoeFusedGroupedGEMMBlock`: Extends the base block to use Triton grouped GEMM kernels with configurable permutation fusion (permute_x in first GEMM prologue, permute_y in second GEMM epilogue), autotuning support, and manual kernel configuration

The forward pass follows the pattern: router → permute → first_gemm (gate_up_proj) → act_and_mul → second_gemm (down_proj) → unpermute → merge topk weights.

**Significance:** Essential for testing Triton kernels against known-good torch implementations. Qwen3 uses top-8 routing across 128 experts with softmax-based routing and topk normalization, making it a challenging test case for grouped GEMM optimization.
