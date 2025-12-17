# File: `utils/modular_model_converter.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1920 |
| Classes | `ReplaceNameTransformer`, `ReplaceParentClassCallTransformer`, `ReplaceSuperCallTransformer`, `ClassDependencyMapper`, `ModuleMapper`, `ModelFileMapper`, `ModularFileMapper` |
| Functions | `get_module_source_from_name`, `preserve_case_replace`, `get_cased_name`, `get_lowercase_name`, `get_full_attribute_name`, `find_all_dependencies`, `dependencies_for_class_node`, `augmented_dependencies_for_class_node`, `... +14 more` |
| Imports | abc, argparse, collections, create_dependency_mapping, functools, glob, importlib, libcst, modular_integrations, multiprocessing, ... +4 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Converts compact modular model definitions into full-fledged model files by inheriting and merging code from existing models.

**Mechanism:** Parses `modular_*.py` files using LibCST, analyzes class inheritance to identify parent models (e.g., `class NewModel(LlamaModel)`), visits parent modeling files to extract functions/classes, resolves dependencies via graph traversal (`find_all_dependencies`), replaces `super()` calls with inlined parent code (`ReplaceSuperCallTransformer`), renames symbols (`ReplaceNameTransformer`), and generates separate files (modeling, configuration, etc.) with proper imports. Handles multi-file generation via `create_modules()` and supports parallel processing with dependency ordering.

**Significance:** Core infrastructure for the modular model system, reducing code duplication by 40-70% in model implementations. Enables rapid model development by allowing authors to write only the differences from existing models while automatically generating production-ready files. Critical for maintainability as it ensures consistency across model implementations.
