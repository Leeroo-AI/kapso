# File: `tests/test_executorch.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 129 |
| Classes | `ExecutorchTest` |
| Imports | torch, transformers, unittest |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests ExecuTorch export capabilities for decoder-only language models with static and hybrid caching mechanisms.

**Mechanism:** The `ExecutorchTest` class validates three ExecuTorch exportable module types: `TorchExportableModuleWithStaticCache` for static KV-cache, `TorchExportableModuleWithHybridCache` for mixed attention patterns (supporting sliding window), and `TorchExportableModuleForDecoderOnlyLM` for general decoder models. Tests verify that exported modules produce outputs matching eager mode execution and that export validation correctly handles input_ids vs inputs_embeds. Uses a tiny random LLaMA model for testing with both token IDs and embedding inputs.

**Significance:** Ensures transformers models can be correctly exported to ExecuTorch format for on-device deployment. Critical for mobile and embedded use cases where models need to run efficiently on edge devices. Validates that optimized cache implementations maintain numerical accuracy after export.
