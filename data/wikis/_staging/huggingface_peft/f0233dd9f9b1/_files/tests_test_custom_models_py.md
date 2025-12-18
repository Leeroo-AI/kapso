# File: `tests/test_custom_models.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 6350 |
| Classes | `MLP`, `MLPWithGRU`, `MLP_LayerNorm`, `MLP2`, `Block`, `DeepMLP`, `ModelEmbConv1D`, `ModelEmbWithEmbeddingUtils`, `ModelConv1D`, `ModelConv1DBigger`, `ModelConv2D`, `ModelConv2D2`, `ModelConv2D1x1`, `ModelConv2DGroups`, `ModelConv2DGroups2`, `ModelConv1DKernel1`, `ModelConv3D`, `ModelMha`, `_LinearUsingParameter`, `MlpUsingParameters`, `MockTransformerWrapper`, `TestPeftCustomModel`, `TestMultiRankAdapter`, `TestLayerRepr`, `TestMultipleActiveAdapters`, `TestRequiresGrad`, `TestMixedAdapterBatches`, `TestDynamicDispatch`, `MyModel`, `EmbModel`, `MLP2`, `MLP2`, `MLP2`, `MLP2`, `MyModule`, `MyLora`, `MyModel`, `MyModel`, `MyLora` |
| Imports | contextlib, copy, functools, os, peft, platform, pytest, re, safetensors, shutil, ... +6 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for PEFT methods on custom model architectures

**Mechanism:** Extensive tests covering LoRA, AdaLora, BOFT, IA3, OFT, LoHa, LoKr, LN-Tuning, VeRA, and other adapters on custom models with Linear, Conv1D, Conv2D, Conv3D, Embedding, GRU, and MHA layers. Tests adapter merging, multi-rank adapters, layer representation, multiple active adapters, gradient management, mixed adapter batches, and dynamic dispatch

**Significance:** Test coverage for comprehensive adapter functionality on diverse layer types
