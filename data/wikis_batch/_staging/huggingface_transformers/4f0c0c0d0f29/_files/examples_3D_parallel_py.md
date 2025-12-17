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

**Purpose:** Demonstrates training transformer models using 3D parallelism combining Tensor Parallelism (TP), Data Parallelism (DP), and Context Parallelism (CP) with PyTorch distributed training.

**Mechanism:** Utilizes PyTorch's DeviceMesh to create a 3D parallelization strategy, integrating FSDP for data parallelism, tensor parallelism through device meshes, and context parallelism for sequence sharding. Trains on TinyStories dataset with sequence packing, custom gradient aggregation across parallel dimensions, and distributed checkpoint saving via DCP. Supports WandB logging and configurable parallelism dimensions through environment variables.

**Significance:** Provides a comprehensive example for advanced distributed training techniques essential for scaling large language models. Showcases best practices for combining multiple parallelism strategies, handling gradient synchronization across complex device topologies, and managing checkpoints in distributed environments. Critical reference implementation for users working with multi-GPU training at scale.
