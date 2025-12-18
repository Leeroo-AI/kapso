# File: `utils/modular_integrations.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 184 |
| Classes | `AbsoluteImportTransformer`, `RelativeImportTransformer` |
| Functions | `convert_relative_import_to_absolute`, `convert_to_relative_import` |
| Imports | libcst, os |

## Understanding

**Status:** ✅ Explored

**Purpose:** Converts between relative and absolute import statements in Python code for cross-library compatibility.

**Mechanism:** Uses libcst (Concrete Syntax Tree) library to parse and transform ImportFrom nodes. For relative-to-absolute conversion, counts leading dots to determine nesting level, resolves file path to construct full module path including package name. For absolute-to-relative conversion, calculates depth from package root and strips package prefix from module path. Supports special handling for namespace packages like optimum.

**Significance:** Critical infrastructure for the modular model system that enables code generation for external libraries (e.g., optimum-habana, optimum-intel). Ensures generated model files use appropriate import styles for their target library while maintaining compatibility with transformers' modular architecture.
