# File: `method_comparison/sanitizer.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 100 |
| Functions | `parse_and_filter` |
| Imports | ast, pandas |

## Understanding

**Status:** ✅ Explored

**Purpose:** Secure DataFrame filtering engine that parses user-provided filter expressions into boolean masks without exposing the application to arbitrary code execution vulnerabilities.

**Mechanism:** Uses Python's ast (Abstract Syntax Tree) module to parse filter strings in eval mode, then recursively evaluates the AST nodes to build pandas boolean Series. Supports comparison operators (>, >=, <, <=, ==, !=), membership operators (in, not in), boolean operations (and, or), bitwise operations (&, |), and negation (not). The _evaluate_node() function walks the AST tree, validating that only safe operations are used and that column names exist in the DataFrame, then translates AST nodes into pandas operations.

**Significance:** Critical security component that prevents code injection attacks while still allowing flexible data filtering in the Gradio interface. By avoiding DataFrame.query() (which uses eval() internally), it protects against malicious filter strings like those attempting os.system() calls while maintaining a natural query syntax for users.
