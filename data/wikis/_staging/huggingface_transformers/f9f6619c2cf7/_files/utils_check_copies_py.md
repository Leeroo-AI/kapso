# File: `utils/check_copies.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1044 |
| Functions | `find_block_end`, `split_code_into_blocks`, `find_code_in_transformers`, `replace_code`, `find_code_and_splits`, `get_indent`, `run_ruff`, `stylify`, `... +5 more` |
| Imports | argparse, collections, glob, os, re, subprocess, transformers |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Validates and automatically fixes code marked with `# Copied from` comments to ensure copied code stays synchronized with its source across the codebase.

**Mechanism:** The script scans all Python files for `# Copied from` comments, which indicate code blocks that should mirror another function/class. For each marked section, it locates the source code, applies any specified transformations (e.g., `with BertModel->RobertaModel` to handle name changes), and compares it against the actual code. The comparison uses AST-based code splitting into blocks (functions, classes, methods) with support for stylification via ruff formatting. Special markers like `# Ignore copy` allow intentional deviations. The script also validates model lists in README files across different language versions for consistency.

**Significance:** This automation is critical for maintaining a large codebase with many similar model implementations. Many models share substantial code (e.g., BERT variants), and this system allows DRY principles while enabling model-specific customizations. It prevents drift where copied code diverges from its source, catching bugs that would otherwise spread across models. The tool is integral to `make fix-copies` and `make repo-consistency` workflows.
