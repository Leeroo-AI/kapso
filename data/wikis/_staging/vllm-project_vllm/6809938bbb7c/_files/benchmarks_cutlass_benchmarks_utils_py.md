# File: `benchmarks/cutlass_benchmarks/utils.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 100 |
| Functions | `to_fp8`, `to_int8`, `to_bf16`, `to_fp16`, `make_rand_tensors`, `prune_to_2_4`, `make_rand_sparse_tensors`, `make_n_rand_sparse_tensors` |
| Imports | collections, torch, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides utility functions for generating test tensors for CUTLASS benchmark scripts.

**Mechanism:** Implements dtype conversion functions (to_fp8, to_int8, to_bf16, to_fp16) with proper clamping and rounding. make_rand_tensors() generates random dense matrices for benchmarking. prune_to_2_4() implements structured sparsity by keeping only the top-2 absolute values in each group of 4 elements, creating 2:4 sparse tensors. make_rand_sparse_tensors() combines generation and pruning, then calls cutlass_sparse_compress() to create compressed representations and metadata. make_n_rand_sparse_tensors() batches tensor generation for multi-tensor benchmarks.

**Significance:** Core infrastructure for CUTLASS benchmarking. The 2:4 sparsity pattern is hardware-accelerated on NVIDIA Ampere+ GPUs, making prune_to_2_4() essential for sparse kernel validation. Ensures consistent test data generation across different benchmark scripts (sparse_benchmarks.py, w8a8_benchmarks.py). The compressed format matches CUTLASS library expectations for sparse matrix operations.
