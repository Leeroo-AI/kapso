# File: `unsloth/kernels/moe/grouped_gemm/kernels/forward.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 265 |
| Imports | grouped_gemm, torch, triton |

## Understanding

**Status:** ✅ Explored

**Purpose:** Triton kernel for grouped GEMM forward pass

**Mechanism:** Implements `_grouped_gemm_forward_kernel` that performs batched matrix multiplications for multiple experts, each processing variable-sized token groups. Uses persistent thread blocks that iterate over experts and tiles. Supports optional fusions: PERMUTE_X (gather tokens on load), PERMUTE_Y (scatter results on store), FUSE_MUL_POST (multiply by routing weights). TMA support for efficient memory access on Hopper+ GPUs.

**Significance:** Core forward pass kernel for MoE layers. Eliminates separate permutation kernels by fusing gather/scatter operations into the GEMM, significantly reducing memory traffic and improving performance for MoE models like Llama4 and Qwen3.

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
