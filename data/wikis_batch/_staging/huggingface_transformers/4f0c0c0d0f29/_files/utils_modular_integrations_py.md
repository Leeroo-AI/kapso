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

**Purpose:** Converts between relative and absolute import statements in Python AST for modular model integration.

**Mechanism:** Uses LibCST to transform import nodes: `convert_relative_import_to_absolute()` resolves relative imports (e.g., `from ..utils import X`) to full paths (e.g., `from transformers.models.llama.utils import X`) based on file path and package name. `convert_to_relative_import()` performs the inverse. Provides transformer classes `AbsoluteImportTransformer` and `RelativeImportTransformer` for batch processing. Handles namespace packages like `optimum.habana`.

**Significance:** Critical infrastructure for the modular model system, enabling code to be seamlessly moved between transformers and external libraries (e.g., optimum) while maintaining correct import paths. Essential for external library integration and modular model file generation.
