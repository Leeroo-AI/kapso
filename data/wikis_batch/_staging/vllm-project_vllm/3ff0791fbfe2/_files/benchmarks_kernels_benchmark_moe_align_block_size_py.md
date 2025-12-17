# File: `benchmarks/kernels/benchmark_moe_align_block_size.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 87 |
| Functions | `get_topk_ids`, `benchmark` |
| Imports | argparse, itertools, torch, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Benchmarks MoE block alignment kernel

**Mechanism:** Uses Triton's benchmarking framework to measure performance of moe_align_block_size function. Tests across multiple configurations: num_tokens (1-4096), num_experts (16-512), topk (1-8), and expert parallelism sizes (1-8). Creates random topk_ids and optional expert mapping for expert parallelism. Measures latency at different quantiles (20%, 50%, 80%). Designed to work with Triton's perf_report decorator for automatic plotting and comparison.

**Significance:** Validates performance of the MoE block alignment operation, which is crucial for preparing token distributions to experts before MoE computation. Important for ensuring efficient memory access patterns in MoE layers. Helps tune block_size parameter (default 256) and validates expert parallelism support.
