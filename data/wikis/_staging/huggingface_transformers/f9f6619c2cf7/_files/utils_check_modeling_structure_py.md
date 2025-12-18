# File: `utils/check_modeling_structure.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 150 |
| Functions | `iter_modeling_files`, `colored_error_message`, `full_name`, `check_init_weights`, `is_self_method_call`, `is_super_method_call`, `check_post_init`, `main` |
| Imports | ast, pathlib, rich, sys |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Enforces critical structural conventions in modeling files using AST analysis to catch common implementation mistakes.

**Mechanism:** Parses all `modeling_*.py` and `modular_*.py` files with Python's AST module to perform two key checks: (1) `_init_weights` methods must use `init.*()` functions instead of in-place tensor operations to properly set initialization flags on parameters, and (2) `PreTrainedModel` subclasses must call `self.post_init()` at the end of `__init__` to ensure proper model setup.

**Significance:** Prevents subtle bugs that would break model initialization and weight management. The `_init_weights` check ensures the lazy initialization system works correctly, while the `post_init` check ensures models are properly configured before use.
