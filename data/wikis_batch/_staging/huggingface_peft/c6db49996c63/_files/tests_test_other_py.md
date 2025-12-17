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

**Purpose:** Tests for miscellaneous PEFT functionality edge cases

**Mechanism:** Tests modules_to_save behavior with various container types (ModuleDict, ParameterList), attribute access on wrapped modules, substring bugs in adapter names, targeting restrictions (preventing adapters from targeting themselves or auxiliary wrappers), no-split modules detection, and tied weights handling

**Significance:** Ensures PEFT handles edge cases correctly including special module types, prevents common user errors, and maintains compatibility with transformers' model architecture features like tied embeddings
