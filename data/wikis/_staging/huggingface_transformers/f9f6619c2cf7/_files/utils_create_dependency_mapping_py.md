# File: `utils/create_dependency_mapping.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 116 |
| Functions | `topological_sort`, `is_model_import`, `extract_model_imports_from_file`, `find_priority_list` |
| Imports | ast, collections, re |

## Understanding

**Status:** ✅ Explored

**Purpose:** Analyzes modular model files to build a dependency graph and determine the correct processing order based on model inheritance relationships.

**Mechanism:** Uses AST parsing to extract model-specific imports from modular files (imports from other model's modeling/config/tokenization files). Constructs a dependency graph where nodes are models and edges represent inheritance dependencies. Performs topological sorting to organize models into levels where each level contains models that only depend on models from previous levels.

**Significance:** Essential for the modular architecture where models can inherit from other models (e.g., Gemma2 inherits from Gemma). Ensures modular conversion and testing happens in the correct order, so base models are processed before derived models that depend on them.
