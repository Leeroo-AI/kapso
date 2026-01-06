# File: `tests/test_other.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 624 |
| Classes | `ModelWithModuleDict`, `ModelWithModuleList`, `ModelWithParameterDict`, `ModelWithParameterList`, `TestModulesToSaveAttributeAccess`, `TestModulesToSaveNameSubstringBug`, `TestTargetingAuxiliaryTrainingWrapper`, `TestAdapterTargeting`, `TestGetNoSplitModules`, `TestGetModuleNamesTiedWithEmbedding`, `MLP`, `MyModule`, `PlainModel`, `NestedModel`, `M` |
| Functions | `test_modules_to_save_targets_module_dict_raises`, `test_get_peft_model_revision_warning`, `test_load_multiple_adapters_different_modules_to_save` |
| Imports | contextlib, copy, peft, pytest, testing_utils, torch, transformers, unittest |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for miscellaneous PEFT functionality

**Mechanism:** Tests modules_to_save with ModuleDict/ModuleList/ParameterDict/ParameterList (should raise errors), revision warnings, tied modules, and no_split_modules detection for various model architectures

**Significance:** Test coverage for edge cases and utility functions
