# File: `tests/test_gpu_examples.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 1148+ |
| Classes | Multiple test classes |
| Imports | accelerate, datasets, gc, importlib, itertools, numpy, os, packaging, parameterized, pathlib, peft, pytest, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for GPU-specific examples and advanced features

**Mechanism:** Comprehensive GPU tests including multi-GPU, FSDP, quantization (bitsandbytes, AQLM, AWQ, GPTQ, EETQ, HQQ, TorchAO), LoftQ, EVA, ARROW, hotswap adapter functionality, NFQuantizer, and distributed training scenarios with various PEFT methods

**Significance:** Test coverage for advanced GPU features and quantization integrations
