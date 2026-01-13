# File: `unsloth/kernels/moe/tests/moe_utils.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 507 |
| Classes | `ForwardResult`, `BackwardResult`, `Qwen3MoeFusedGroupedGEMMBlock` |
| Functions | `rebind_experts_to_shared_buffer`, `get_expert_metadata`, `clone_experts`, `check_down_proj_grad`, `check_gate_up_proj_grad`, `check_gate_grad`, `check_wgrad`, `check_tensor_allclose`, `... +6 more` |
| Imports | dataclasses, grouped_gemm, huggingface_hub, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides MoE-specific test utilities for comparing HuggingFace reference implementations against Triton grouped GEMM implementations, with extensive gradient checking capabilities.

**Mechanism:** Implements `ForwardResult` and `BackwardResult` dataclasses to capture outputs and gradients for comparison. Provides weight manipulation functions: `rebind_experts_to_shared_buffer()` consolidates per-expert weights into contiguous buffers, `clone_experts()` extracts weights from HF MoE blocks. Gradient checking functions (`check_down_proj_grad`, `check_gate_up_proj_grad`, `check_gate_grad`, `check_expert_grads`) compare gradients element-wise with tolerance assertions. The `Qwen3MoeFusedGroupedGEMMBlock` class extends `Qwen3MoeGroupedGEMMBlock` to use Triton grouped GEMM kernels instead of torch-native implementations, supporting configurable permute_x/permute_y fusion flags and autotuning. The forward pass orchestrates routing, token permutation, two grouped GEMMs (gate-up projection and down projection), SiLU activation, and final unpermutation with topk weight application.

**Significance:** Critical testing infrastructure that enables validation of the optimized Triton MoE kernels against HuggingFace's reference implementation. The detailed intermediate result tracking (`GroupedGEMMResult`) facilitates debugging of complex fused operations.
