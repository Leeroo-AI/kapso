# File: `benchmarks/kernels/weight_shapes.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 104 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines weight shapes with TP split information

**Mechanism:** Enhanced weight shape dictionary with tensor parallelism split dimension. Format: ([K, N], TP_SPLIT_DIM) where TP_SPLIT_DIM indicates which dimension is sharded (0=K, 1=N). Includes shapes for Mistral-7B, Llama-2-7b/13b/70b, Llama-3-8b, and DeepSeek-V2-Lite. Enables automatic calculation of sharded dimensions for different TP sizes. More sophisticated than benchmark_shapes.py as it encodes sharding semantics.

**Significance:** Critical for benchmarks that need to account for tensor parallelism. Enables testing realistic sharded workloads without manual dimension calculation. Essential for LoRA, Machete, and other benchmarks that support multi-GPU configurations. Helps validate kernel performance under actual distributed inference conditions. More maintainable than hardcoding TP variants.
