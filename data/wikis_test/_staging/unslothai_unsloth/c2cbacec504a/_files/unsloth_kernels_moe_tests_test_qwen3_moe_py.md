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

**Purpose:** Integration tests for Qwen3 MoE block comparing multiple implementations.

**Mechanism:** Tests forward/backward passes with reference MoE utilities, supports permutation fusion options and autotuning, parametrized for different configurations.

**Significance:** Validates Qwen3 MoE implementation end-to-end ensuring correctness across different kernel configurations.
