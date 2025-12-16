# File: `scripts/enforce_kwargs_spacing.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 179 |
| Functions | `enforce_spacing`, `remove_redundant_passes`, `process_file`, `main` |
| Imports | __future__, argparse, ast, collections, io, pathlib, sys, tokenize |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Code formatting script that enforces Unsloth's style convention of spaces around `=` in keyword arguments.

**Mechanism:**
- `enforce_spacing()`: Uses Python tokenizer to find all `=` operators and ensures they have spaces on both sides
- Tracks column offsets as spaces are inserted to maintain correct positions
- `remove_redundant_passes()`: Uses AST to find `pass` statements that are unnecessary (in blocks with other statements)
- `process_file()`: Reads file, applies both transformations, writes back if changed
- Preserves file encoding and handles edge cases (multi-line, invalid syntax)
- Skips modifying itself to avoid self-edit loops

**Significance:** Development tooling to maintain consistent code style. Unsloth uses `param = value` rather than PEP-8's `param=value` for keyword arguments, which this script enforces.
