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

**Purpose:** Validates and auto-fixes docstrings to ensure they match function/class signatures, with special support for `@auto_docstring` decorated objects that inherit documentation from centralized sources.

**Mechanism:** The script operates in two modes: (1) For regular docstrings, it uses introspection to compare documented arguments against function signatures, checking that defaults match and all parameters are documented with proper types and descriptions. (2) For `@auto_docstring` decorated objects, it parses the AST to find decorated functions/classes, generates docstrings by combining inherited documentation from `ModelArgs`/`ImageProcessorArgs` with custom overrides, and removes redundant documentation that duplicates the inherited sources. The script can detect missing arguments, fill templates, validate argument ordering, and handle special cases like ModelOutput dataclasses.

**Significance:** Docstring consistency is crucial for auto-generated API documentation and IDE tooltips. This tool ensures documentation stays synchronized with code as signatures evolve, reducing manual maintenance burden. The `@auto_docstring` system is particularly important as it allows model implementations to inherit standard parameter documentation while only documenting model-specific differences, dramatically reducing documentation duplication across 100+ models while maintaining accuracy.
