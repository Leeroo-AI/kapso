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

**Purpose:** Converts modular model definitions into traditional single-file implementations by merging inherited code and resolving dependencies.

**Mechanism:** Parses modular_*.py files using libcst, visits imported modeling files to build dependency graphs of classes/functions/assignments, handles class inheritance by merging parent class code with child overrides, unravels super() calls by inlining parent method code, tracks recursive dependencies to include all needed components, and automatically determines correct import statements and file types (modeling, configuration, processing, etc.) for generated outputs.

**Significance:** Core infrastructure for the modular model architecture that enables DRY (Don't Repeat Yourself) model development. Allows developers to write concise modular definitions inheriting from existing models (e.g., 200 lines instead of 2000), while automatically generating complete standalone files for distribution. Reduces code duplication by 80-90% while maintaining backward compatibility and supporting multi-file model architectures.
