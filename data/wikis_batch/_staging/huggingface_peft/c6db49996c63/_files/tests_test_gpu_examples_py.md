# File: `tests/test_gpu_examples.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 5682 |
| Classes | `DataCollatorSpeechSeq2SeqWithPadding`, `PeftBnbGPUExampleTests`, `PeftGPTQGPUTests`, `TestOffloadSave`, `TestPiSSA`, `TestOLoRA`, `TestLoftQ`, `MultiprocessTester`, `MixedPrecisionTests`, `PeftAqlmGPUTests`, `PeftHqqGPUTests`, `PeftAwqGPUTests`, `PeftEetqGPUTests`, `PeftTorchaoGPUTests`, `SimpleModel`, `SimpleConv2DModel`, `TestAutoCast`, `TestFSDPWrap`, `TestBOFT`, `TestPTuningReproducibility`, `TestLowCpuMemUsageDifferentDevices`, `TestEvaInitializationGPU`, `TestALoRAInferenceGPU`, `TestPrefixTuning`, `TestHotSwapping`, `TestArrowQuantized`, `TestDtypeAutocastBnb`, `OptimizerStepCallback`, `OptimizerStepCallback`, `OptimizerStepCallback`, `MyModule` |
| Imports | accelerate, collections, copy, dataclasses, datasets, gc, importlib, itertools, numpy, os, ... +13 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for GPU-specific PEFT functionality including quantization, distributed training, and special features.

**Mechanism:** Comprehensive test suite covering BitsAndBytes quantization (8-bit, 4-bit), GPTQ, AQLM, HQQ, AWQ, EETQ, and TorchAO quantization methods. Tests include FSDP wrapping, mixed precision training, autocast behavior, prefix tuning, P-tuning reproducibility, low CPU memory usage, device mapping, hotswapping adapters, LoftQ initialization, PiSSA, OLoRA, Eva initialization, Arrow quantization, and multi-adapter scenarios. Uses multiple test classes for different features.

**Significance:** Critical validation of PEFT with GPU-accelerated features and various quantization backends. This massive test suite (5682 lines) ensures PEFT works correctly with memory-efficient training techniques, distributed setups, and different hardware acceleration methods essential for practical large model fine-tuning.
