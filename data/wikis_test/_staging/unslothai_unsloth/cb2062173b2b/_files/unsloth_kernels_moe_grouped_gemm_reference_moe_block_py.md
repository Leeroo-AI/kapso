# File: `unsloth/kernels/moe/grouped_gemm/reference/moe_block.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 161 |
| Classes | `Qwen3MoeFusedGroupedGEMMBlock` |
| Imports | grouped_gemm, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides a cleaner, production-oriented implementation of Qwen3 MoE block using Triton grouped GEMM kernels, distinct from the more verbose test-oriented version in qwen3_moe.py.

**Mechanism:** Implements `Qwen3MoeFusedGroupedGEMMBlock` as an alternative to the test version, with similar functionality but less verbose debugging output. The class:
- Inherits from `Qwen3MoeGroupedGEMMBlock`
- Accepts configuration for permutation fusion (permute_x/permute_y), autotuning, and manual kernel configs
- Provides `from_hf()` class method to construct from HuggingFace models
- Implements forward pass using Triton grouped GEMM kernels with optional permutation fusion in prologue/epilogue
- Conditionally permutes inputs/outputs based on fusion settings
- Returns `GroupedGEMMResult` with all intermediate values

Note in docstring: "NOT to be used for production as it contains many extra checks and saves all intermediate results for debugging" - suggests this is still a reference/test implementation despite the cleaner interface.

**Significance:** Provides a simpler reference implementation focusing on the Triton kernel integration without the extensive test scaffolding. Serves as a bridge between test code and production usage patterns.
