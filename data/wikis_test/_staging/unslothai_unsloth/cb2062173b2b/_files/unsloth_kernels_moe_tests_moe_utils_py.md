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

**Purpose:** Provides specialized testing utilities for Qwen3 MoE models, including expert weight management, forward/backward execution, gradient validation, and result comparison functions.

**Mechanism:** Implements comprehensive test support organized into several categories:

1. Weight Management:
   - `rebind_experts_to_shared_buffer()`: Reorganizes expert weights into contiguous buffers for efficient access
   - `clone_experts()`: Extracts and stacks expert weights from HuggingFace models
   - `get_expert_metadata()`: Fetches safetensors metadata from HuggingFace Hub

2. Result Storage:
   - `ForwardResult`: Stores output, router logits, input X, and optional grouped GEMM intermediate results
   - `BackwardResult`: Stores gradients for X, gate, gate_proj, up_proj, down_proj

3. Execution Helpers:
   - `run_forward()`: Executes forward pass with proper gradient setup, handles both HF and grouped GEMM blocks
   - `run_backward()`: Runs backward pass and extracts all gradients from either HF or grouped GEMM implementations

4. Validation Functions:
   - `check_fwd()`: Validates output and router logits between implementations
   - `check_grads()`: Validates X.grad and gate.grad, then calls expert grad checkers
   - `check_expert_grads()`: Validates gradients for all expert projections across all experts
   - `check_grouped_gemm_results()`: Validates all intermediate results (token counts, routing indices, first/second GEMM, etc.) with special handling for permute_y flag
   - Individual gradient checkers: `check_down_proj_grad()`, `check_gate_up_proj_grad()`, `check_gate_grad()`

5. Reference Implementation:
   - `Qwen3MoeFusedGroupedGEMMBlock`: Test-oriented version with extensive intermediate result tracking

**Significance:** Critical for end-to-end testing of Qwen3 MoE implementations. Enables detailed validation of every component (forward outputs, backward gradients, intermediate states) between HuggingFace reference, torch grouped GEMM, and Triton grouped GEMM implementations. The granular validation functions help pinpoint exactly where numerical differences occur.
