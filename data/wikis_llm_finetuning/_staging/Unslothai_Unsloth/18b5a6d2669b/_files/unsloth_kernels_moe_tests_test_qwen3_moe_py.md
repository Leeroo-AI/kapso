# File: `unsloth/kernels/moe/tests/test_qwen3_moe.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 273 |
| Functions | `model_id`, `config`, `annotated_context`, `test_qwen3_moe` |
| Imports | argparse, contextlib, grouped_gemm, moe_utils, pytest, torch, transformers |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Integration tests for Qwen3 MoE layer implementation with grouped GEMM acceleration.

**Mechanism:** Parametrized pytest tests comparing three implementations: HF Qwen3MoeSparseMoeBlock (reference), Qwen3MoeGroupedGEMMBlock (torch), and Qwen3MoeFusedGroupedGEMMBlock (Triton). Tests forward outputs, backward gradients (X, gate, gate_proj, up_proj, down_proj), and intermediate results (first_gemm, intermediate, second_gemm) across sequence lengths, dtypes, permutation modes, and autotuning. Uses moe_utils helpers (run_forward, run_backward, check_fwd, check_grads, check_grouped_gemm_results). Note: recommends running as module (python -m tests.test_qwen3_moe) rather than pytest due to triton/autotuning interactions.

**Significance:** Critical validation for Qwen3 support - models with 128 experts and top-8 routing represent the most challenging MoE workload. The intermediate result checking helps diagnose where numerical divergence occurs in the fusion pipeline. The test suite stress-tests load balancing, memory bandwidth, and numerical stability. Success here proves the grouped GEMM approach scales to extreme MoE configurations.
