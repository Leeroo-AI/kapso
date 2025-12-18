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

**Purpose:** Benchmarks moe_align_block_size operation performance across different token counts, expert counts, top-k values, and expert parallelism configurations.

**Mechanism:** Uses Triton benchmarking framework to measure performance of aligning MoE tokens and experts into block-sized chunks. Tests various configurations with/without expert parallelism and expert mapping.

**Significance:** Critical for optimizing MoE token-to-expert routing alignment which impacts kernel launch efficiency and memory access patterns in MoE layers.
