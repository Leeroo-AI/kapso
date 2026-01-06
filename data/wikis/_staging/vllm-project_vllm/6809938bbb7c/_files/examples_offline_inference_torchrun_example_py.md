# File: `examples/offline_inference/torchrun_example.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 76 |
| Imports | torch, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Basic example of using torchrun to launch vLLM with tensor parallelism across multiple GPUs.

**Mechanism:** Launched via torchrun with tensor_parallel_size matching world size. Uses distributed_executor_backend="mp" for multiprocessing coordination. Each process initializes its portion of the tensor-parallel model, with torch.distributed handling communication.

**Significance:** Shows minimal torchrun integration for tensor parallelism. Provides foundation for understanding distributed execution with torchrun before exploring more complex patterns like data parallelism.
