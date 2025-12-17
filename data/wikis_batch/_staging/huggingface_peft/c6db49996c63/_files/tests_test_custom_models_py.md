# File: `tests/test_custom_models.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 6350 |
| Classes | `MLP`, `MLPWithGRU`, `MLP_LayerNorm`, `MLP2`, `Block`, `DeepMLP`, `ModelEmbConv1D`, `ModelEmbWithEmbeddingUtils`, `ModelConv1D`, `ModelConv1DBigger`, `ModelConv2D`, `ModelConv2D2`, `ModelConv2D1x1`, `ModelConv2DGroups`, `ModelConv2DGroups2`, `ModelConv1DKernel1`, `ModelConv3D`, `ModelMha`, `_LinearUsingParameter`, `MlpUsingParameters`, `MockTransformerWrapper`, `TestPeftCustomModel`, `TestMultiRankAdapter`, `TestLayerRepr`, `TestMultipleActiveAdapters`, `TestRequiresGrad`, `TestMixedAdapterBatches`, `TestDynamicDispatch`, `MyModel`, `EmbModel`, `MLP2`, `MLP2`, `MLP2`, `MLP2`, `MyModule`, `MyLora`, `MyModel`, `MyModel`, `MyLora` |
| Imports | contextlib, copy, functools, os, peft, platform, pytest, re, safetensors, shutil, ... +6 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for PEFT with custom (non-transformers) models.

**Mechanism:** Contains extensive test classes with custom model definitions (MLP, CNN, GRU, attention variants) testing: basic PEFT operations, multi-rank adapters, layer representations, multiple active adapters, requires_grad behavior, mixed adapter batches, dynamic dispatch, Conv1D/Conv2D/Conv3D support, embedding handling, module-to-save functionality, custom layer implementations, and adapter composition. Covers 20+ PEFT methods across various architectures.

**Significance:** Critical for validating PEFT works beyond transformers models, ensuring it can adapt custom architectures with different layer types (Linear, Conv, Embedding, GRU) and supporting diverse research use cases.
