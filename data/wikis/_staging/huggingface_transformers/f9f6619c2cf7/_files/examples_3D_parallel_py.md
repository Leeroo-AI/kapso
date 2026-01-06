# File: `examples/3D_parallel.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 434 |
| Classes | `AppState` |
| Functions | `main`, `all_reduce_grads`, `clip_grad_norm_` |
| Imports | collections, contextlib, datasets, logging, os, torch, transformers, wandb |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates training transformer models using 3D parallelism (Tensor Parallelism, Data Parallelism, and Context Parallelism) with PyTorch's distributed training capabilities.

**Mechanism:** Creates a 3D device mesh with configurable TP_SIZE, DP_SIZE, and CP_SIZE dimensions, loads a causal language model (SmolLM2-1.7B by default) with tensor parallelism support, wraps it with FSDP for data parallelism when needed, and applies context parallelism to sequence dimensions. The training loop processes the TinyStories dataset with packed sequences, using PyTorch's distributed checkpoint (DCP) for saving model state. It includes custom gradient synchronization across the dp_cp mesh dimensions, gradient clipping for DTensor parameters, and comprehensive logging to Weights & Biases. The script handles edge cases like DTensor gradient all-reduce, position_ids creation for context parallel sharding, and uses SDPA kernel selection for attention operations.

**Significance:** Provides a reference implementation for advanced distributed training techniques in the transformers library, showcasing how to combine multiple parallelism strategies for training large language models efficiently. This is critical for users who need to scale training beyond what single-GPU or basic data parallel approaches can handle, demonstrating integration with PyTorch's latest distributed primitives including DeviceMesh, DTensor, and FSDP with NO_SHARD strategy.
