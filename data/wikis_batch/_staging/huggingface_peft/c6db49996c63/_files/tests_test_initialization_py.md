# File: `tests/test_initialization.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 5029 |
| Classes | `TestLoraInitialization`, `TestLokrInitialization`, `TestAdaLoraInitialization`, `TestPromptTuningInitialization`, `TestVeraInitialization`, `TestVBLoraInitialization`, `TestC3AInitialization`, `TestWaveFTInitialization`, `TestRoadInitialization`, `TestDeLoRAInitialization`, `TestGraLoRAInitialization`, `TestNoInfiniteRecursionDeepspeed`, `TestLoadAdapterOfflineMode`, `TestCustomModelConfigWarning`, `TestLowCpuMemUsage`, `TestNamingConflictWarning`, `TestCordaInitialization`, `TestEvaInitialization`, `TestHotSwapping`, `TestScaling`, `TestLoadPeftKeyMapping`, `TestWeightTying`, `MyModule`, `ModelMha`, `ModelConv2DGroups`, `MyModule`, `MyModule`, `MyModule`, `MyModule`, `MLP`, `MLP`, `MLP`, `MyModule`, `MLP`, `MyModule`, `MLP`, `MLP`, `MyModule`, `MyModule`, `MyModule`, `MLP`, `ConvModel`, `FakeConfig`, `Block`, `OldModel`, `Block`, `InnerModel`, `NewModel`, `MyModule`, `CausalLM` |
| Functions | `test_from_pretrained_missing_keys_warning`, `test_import_peft_type_to_model_mapping_deprecation_warning` |
| Imports | collections, contextlib, copy, datasets, huggingface_hub, itertools, math, peft, platform, pytest, ... +8 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for adapter initialization, weight loading, scaling, and configuration across all PEFT methods.

**Mechanism:** Contains comprehensive test classes for various PEFT adapters (LoRA, LoKr, AdaLoRA, VeRA, VBLoRA, C3A, WaveFT, RoAD, DeLoRA, GraLoRA, CorDA, EVA). Tests weight initialization schemes (kaiming, gaussian, orthogonal, PiSSA, OLoRA), scaling factors (standard and RSLoRA), DoRA variants, rank/alpha patterns, low CPU memory usage, adapter hotswapping, offline mode, model compilation, and weight tying. Uses statistical tests to verify initialization distributions.

**Significance:** Critical test suite ensuring correct initialization and configuration of all adapter types, proper weight handling during save/load, and compatibility with various optimization techniques and deployment scenarios.
