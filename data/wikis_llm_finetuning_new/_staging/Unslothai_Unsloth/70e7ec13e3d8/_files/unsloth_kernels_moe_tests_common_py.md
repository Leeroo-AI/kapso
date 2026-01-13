# File: `unsloth/kernels/moe/tests/common.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 336 |
| Classes | `DataConfig`, `ModelConfig`, `GroupedGEMMTestConfig` |
| Functions | `print_delimiter`, `delimiter_context`, `make_inputs`, `assert_equal`, `assert_close`, `assert_indx_equal`, `get_kernel_test_configs`, `remove_feature_flags` |
| Imports | contextlib, dataclasses, grouped_gemm, itertools, torch |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides common test infrastructure, configurations, and assertion utilities for testing grouped GEMM kernels in MoE (Mixture of Experts) models.

**Mechanism:** Defines dataclasses for test configuration (`DataConfig`, `ModelConfig`, `GroupedGEMMTestConfig`) that specify model dimensions, sequence lengths, dtypes, and expert counts. Provides `make_inputs()` to generate random test tensors (X, W, scores) for grouped GEMM operations. Includes tolerance-aware assertion functions (`assert_close`, `assert_equal`) that handle numerical precision differences across dtypes (bfloat16, float16, float32). The `get_kernel_test_configs()` function generates combinations of kernel configurations (block sizes, TMA options, permutation flags) for both forward and backward passes, with pruning via `remove_feature_flags()`. Pre-defines model configs for LLAMA4 (5120x8192, 16 experts, topk=1, sigmoid routing) and QWEN (2048x768, 128 experts, topk=8).

**Significance:** Core testing utility that standardizes test setup across all MoE kernel tests. The tolerance handling and comprehensive kernel config generation are critical for validating Triton kernel correctness against PyTorch reference implementations.
