# File: `examples/offline_inference/torchrun_dp_example.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 151 |
| Functions | `parse_args` |
| Imports | argparse, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates launching vLLM with data parallelism using torchrun for distributed execution across multiple GPUs or nodes.

**Mechanism:** Uses torchrun launcher to spawn multiple processes with distributed_executor_backend="mp". Each process runs independent vLLM instance with data_parallel_size configuration. Coordinates prompt distribution across replicas and aggregates results, leveraging torch.distributed for inter-process communication.

**Significance:** Shows how to use torchrun (PyTorch's distributed launcher) for multi-node or multi-GPU data parallel inference. Alternative to Ray-based parallelism, useful for environments where torchrun is preferred.
