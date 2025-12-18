# File: `method_comparison/sanitizer.py`

**Category:** security utility

| Property | Value |
|----------|-------|
| Lines | 101 |
| Functions | `_evaluate_node`, `parse_and_filter` |
| Imports | ast, pandas |

## Understanding

**Status:** Explored

**Purpose:** Provides secure DataFrame filtering using AST (Abstract Syntax Tree) parsing to avoid arbitrary code execution vulnerabilities that pandas.DataFrame.query() exposes.

**Mechanism:**
- `parse_and_filter()`: Main entry point that:
  - Parses filter string into an AST using Python's ast.parse() in 'eval' mode
  - Validates syntax and calls recursive evaluation
  - Returns a boolean mask Series for DataFrame filtering

- `_evaluate_node()`: Recursive AST evaluator that:
  - Handles comparison operations: >, >=, <, <=, ==, !=, in, not in
  - Validates that left side is a column name and right side is a literal
  - Supports boolean operations: and, or (both keyword and bitwise & |)
  - Supports negation with 'not' operator
  - Maps AST node types to pandas operations
  - Rejects chained comparisons and unsupported operators
  - Only allows literal values (numbers, strings, lists) on right side of comparisons

Security features:
- Uses ast.literal_eval() to safely evaluate right-hand values
- Restricts expressions to comparisons and boolean logic only
- Prevents function calls, attribute access, and arbitrary code execution
- Validates column names exist in DataFrame

**Significance:** Critical security component that enables user-provided filtering expressions in the web interface without exposing the system to code injection attacks. This is a safer alternative to pandas.query() or eval(), which would allow arbitrary code execution if user input is not trusted.
