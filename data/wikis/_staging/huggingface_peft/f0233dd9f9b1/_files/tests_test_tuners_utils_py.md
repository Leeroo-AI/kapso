# File: `tests/test_tuners_utils.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 2182 |
| Classes | `TestPeftCustomKwargs`, `MLP`, `TestTargetedModuleNames`, `TestTargetedParameterNames`, `TestExcludedModuleNames`, `TestModelAndLayerStatus`, `MockModelConfig`, `MockModelDataclassConfig`, `ModelWithConfig`, `ModelWithDictConfig`, `ModelWithDataclassConfig`, `ModelWithNoConfig`, `TestBaseTunerGetModelConfig`, `TestBaseTunerWarnForTiedEmbeddings`, `TestFindMinimalTargetModules`, `TestRankAndAlphaPattern`, `SmallModel`, `SmallEmbModel`, `LargeModel`, `SmallModel`, `SmallModel`, `SimpleNet`, `InnerModule`, `OuterModule`, `InnerModule`, `OuterModule`, `Inner`, `Middle`, `Outer` |
| Imports | copy, dataclasses, diffusers, parameterized, peft, pytest, re, testing_utils, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for tuner utility functions

**Mechanism:** Tests regex matching for target_modules, layers_to_transform, layers_pattern, _maybe_include_all_linear_layers, check_target_module_exists, inspect_matched_modules, find_minimal_target_modules, and get_model/layer_status

**Significance:** Test coverage for tuner targeting and utility functions
