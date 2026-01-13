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

**Purpose:** End-to-end test suite for validating Triton grouped GEMM MoE implementation against HuggingFace's Qwen3MoeSparseMoeBlock reference for the Qwen3-30B-A3B model architecture.

**Mechanism:** Tests three implementation levels: HF reference (`Qwen3MoeSparseMoeBlock`), torch grouped GEMM reference (`Qwen3MoeGroupedGEMMBlock`), and Triton implementation (`Qwen3MoeFusedGroupedGEMMBlock` from moe_utils). The forward pass validates final outputs via `check_fwd()`, and intermediate results (first_gemm, intermediate activation, second_gemm, hidden_states_unpermute) via `check_grouped_gemm_results()` for detailed debugging. Backward pass validates gradients for: X (input), gate.weight (router), gate_proj, up_proj, and down_proj using `check_grads()`. Tests are parametrized over dtype (bfloat16), sequence length (1024), autotune mode, permute_x, and permute_y. Uses fixtures for model_id and config loading. Includes note about running as module (`python -m tests.test_qwen3_moe`) rather than pytest due to random numerical errors from pytest-triton interactions.

**Significance:** Validates Triton MoE correctness for Qwen3's sparse MoE architecture (128 experts, topk=8). The detailed intermediate checking is particularly valuable for debugging the complex fusion of permutation operations with grouped GEMM kernels.
