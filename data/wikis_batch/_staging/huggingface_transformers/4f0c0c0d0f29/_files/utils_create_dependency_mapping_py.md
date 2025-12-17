# File: `utils/create_dependency_mapping.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 116 |
| Functions | `topological_sort`, `is_model_import`, `extract_model_imports_from_file`, `find_priority_list` |
| Imports | ast, collections, re |

## Understanding

**Status:** ✅ Explored

**Purpose:** Analyzes modular model file dependencies to determine the correct order for model conversions or operations.

**Mechanism:** Uses AST parsing to extract model-specific imports (matching patterns like `models.{model_type}.{file_type}_{model_type}`). Builds a dependency graph showing which modular models import from other modular models, then performs topological sorting to create ordered levels. Returns a list of lists where each level contains models with no dependencies on later levels.

**Significance:** Critical for modular model architecture where models inherit from each other (e.g., Gemma2 from Gemma, GLM4 from Llama4). Ensures models are converted/processed in dependency order so base models exist before derived models that depend on them.
