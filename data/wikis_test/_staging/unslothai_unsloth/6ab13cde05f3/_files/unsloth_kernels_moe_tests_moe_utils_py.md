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

**Purpose:** Qwen3 MoE testing utilities

**Mechanism:** Expert weight extraction, forward/backward running, result checking

**Significance:** Detailed gradient validation helpers
