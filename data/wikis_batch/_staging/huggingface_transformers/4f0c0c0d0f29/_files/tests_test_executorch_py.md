# File: `tests/test_executorch.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 129 |
| Classes | `ExecutorchTest` |
| Imports | torch, transformers, unittest |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests ExecuTorch export functionality for decoder-only language models with static and hybrid caching.

**Mechanism:** ExecutorchTest validates TorchExportableModuleWithStaticCache, TorchExportableModuleWithHybridCache, and TorchExportableModuleForDecoderOnlyLM classes. Tests verify correct forward pass behavior with both input_ids and inputs_embeds, validate cache position handling, ensure exported programs match eager execution outputs, and check proper error handling for invalid input combinations.

**Significance:** Critical for validating PyTorch ExecuTorch integration enabling on-device deployment of transformers models, ensuring exported models maintain numerical accuracy and proper input validation for mobile and edge computing scenarios.
