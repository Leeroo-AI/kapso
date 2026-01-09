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

**Purpose:** MoE-specific testing utilities for validating full model integration and gradient correctness.

**Mechanism:** Provides run_forward() and run_backward() helpers that execute models and collect results (outputs, gradients) in structured dataclasses (ForwardResult, BackwardResult). Implements check_fwd(), check_grads(), and check_grouped_gemm_results() to compare outputs/gradients between implementations with detailed error reporting. Contains check_expert_grads() for per-expert gradient validation and Qwen3MoeFusedGroupedGEMMBlock (test-specific implementation that saves intermediates). Includes rebind_experts_to_shared_buffer() for memory layout testing.

**Significance:** Higher-level testing utilities for integration tests. While test_grouped_gemm.py tests kernel primitives in isolation, these utilities validate that kernels work correctly when integrated into complete MoE layers. The gradient checking is critical for training correctness - subtle bugs in backward passes only manifest here. The per-expert checking helps identify which experts have numerical issues.
