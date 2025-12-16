# File: `unsloth/kernels/moe/grouped_gemm/kernels/forward.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 265 |
| Imports | grouped_gemm, torch, triton |

## Understanding

**Status:** ✅ Explored

**Purpose:** Triton kernel implementation for forward pass grouped GEMM in MoE layers, computing Y = X @ W^T with expert-specific routing.

**Mechanism:**
- `_grouped_gemm_forward_kernel()`: Main Triton JIT kernel that performs batched matrix multiplications across experts
  - Iterates over experts in outer loop, processing variable-sized token groups (m_sizes) per expert
  - For each expert, distributes work tiles across SMs using persistent kernel pattern
  - Inner loop: K-dimension reduction with BLOCK_SIZE_K tiles
  - PERMUTE_X: Loads X in permuted order (token→expert grouping) using gather_indices
  - PERMUTE_Y: Stores Y in permuted order (expert→token restoration) using gather_indices
  - FUSE_MUL_PRE/POST: Optionally multiplies by topk_weights before/after GEMM (inference-only optimization)
  - TMA support: USE_TMA_LOAD_X/W for input loading, USE_TMA_STORE for output storage (Hopper+)
  - Accumulates in float32, converts to output dtype before store
- `_autotuned_grouped_gemm_forward_kernel`: Decorated version with Triton autotuning
  - Autotuning keys: NUM_EXPERTS, NUM_TOKENS, N, K, PERMUTE_X, PERMUTE_Y, FUSE_MUL_POST
  - Uses `get_forward_configs()` and `prune_kernel_configs_fwd()` for configuration space

**Significance:** Core computational kernel for MoE forward pass, enabling efficient batched processing of tokens routed to different experts. The permutation support allows tokens to be grouped by expert for computation efficiency while maintaining correct output ordering. TMA support and autotuning ensure optimal performance across different hardware and workload characteristics.
