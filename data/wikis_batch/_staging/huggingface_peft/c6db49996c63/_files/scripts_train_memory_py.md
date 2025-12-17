# File: `scripts/train_memory.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 276 |
| Functions | `init_accelerator`, `get_data`, `train` |
| Imports | argparse, collections, contextlib, datasets, functools, gc, os, peft, sys, tempfile, ... +4 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Memory profiling and benchmarking script that measures GPU/accelerator memory consumption, training time, model file size, and optionally activation memory for PEFT methods across different configurations.

**Mechanism:** Loads a model with specified dtype (float32/16, bfloat16, int8, int4) via transformers, optionally applies LoRA from a config file, and trains on the ybelkada/english_quotes dataset for a specified number of steps. Tracks peak memory usage via torch.cuda.max_memory_allocated(), measures per-step timing, logs average memory across steps, and saves the adapter to a temp directory to measure file size. When --monitor_tensors is enabled, uses torch.autograd.graph.saved_tensors_hooks to capture all tensors during a single training step, then analyzes their dtypes and shapes to estimate activation memory separate from parameter memory.

**Significance:** Critical benchmarking tool for PEFT method evaluation used to populate the method_comparison results. Enables systematic comparison of memory footprint, training speed, and storage requirements across different PEFT techniques, ranks, and quantization strategies. The activation memory profiling feature helps researchers understand the memory breakdown beyond just parameters, which is essential for optimizing large-scale training.
