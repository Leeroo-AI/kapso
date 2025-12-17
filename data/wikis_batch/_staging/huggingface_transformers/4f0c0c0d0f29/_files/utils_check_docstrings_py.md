# File: `utils/check_docstrings.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1559 |
| Classes | `DecoratedItem` |
| Functions | `find_indent`, `stringify_default`, `eval_math_expression`, `eval_node`, `replace_default_in_arg_description`, `get_default_description`, `find_source_file`, `match_docstring_with_signature`, `... +10 more` |
| Imports | argparse, ast, check_repo, collections, dataclasses, enum, git, glob, inspect, operator, ... +5 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Validates and auto-fixes docstrings to ensure they accurately match function/class signatures, including parameter names, types, and default values.

**Mechanism:** Has two modes: (1) Traditional mode that uses inspect to compare docstring Args sections against function signatures, checking parameter order and defaults. (2) @auto_docstring mode that parses AST to find decorated functions/classes, generates docstrings by combining custom args with inherited args from ModelArgs/ImageProcessorArgs, and updates files directly. Handles complex cases like optional parameters, Enum defaults, and mathematical expressions for defaults.

**Significance:** Essential documentation quality tool that prevents documentation drift as APIs evolve, automatically generates parameter documentation templates for undocumented arguments, and ensures consistency between code signatures and their documentation across 500+ model classes.
