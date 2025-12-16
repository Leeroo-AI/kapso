# File: `unsloth/kernels/moe/tests/moe_utils.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 507 |
| Classes | `ForwardResult`, `BackwardResult`, `Qwen3MoeFusedGroupedGEMMBlock` |
| Functions | `rebind_experts_to_shared_buffer`, `get_expert_metadata`, `clone_experts`, `check_down_proj_grad`, `check_gate_up_proj_grad`, `check_gate_grad`, `check_wgrad`, `check_tensor_allclose`, `check_expert_grads`, `check_grads`, `check_fwd`, `check_grouped_gemm_results`, `run_forward`, `run_backward` |
| Imports | dataclasses, grouped_gemm, huggingface_hub, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides comprehensive testing utilities for validating MoE layer implementations against HuggingFace references.

**Mechanism:**

**Test execution**:
- `run_forward`/`run_backward`: Execute forward/backward passes and collect results
- Returns structured dataclasses (ForwardResult, BackwardResult) with outputs and gradients

**Gradient checking**:
- `check_grads`: Validates all parameter gradients (gate, gate_proj, up_proj, down_proj)
- Expert-level granularity for debugging specific expert failures
- `check_tensor_allclose`: Wrapper with detailed diff reporting

**Intermediate result validation**:
- `check_grouped_gemm_results`: Validates intermediate outputs (first_gemm, intermediate, second_gemm, hidden_states_unpermute)
- Critical for debugging which stage introduces numerical errors

**Weight management**:
- `rebind_experts_to_shared_buffer`: Consolidates expert weights into contiguous buffers
- `clone_experts`: Deep copies expert weights for isolated testing

**Significance:** Comprehensive test harness that enables thorough validation of fused kernels against reference implementations. The detailed gradient checking at expert level is crucial for catching subtle bugs in token routing or permutation logic. Essential for maintaining correctness during kernel optimization.