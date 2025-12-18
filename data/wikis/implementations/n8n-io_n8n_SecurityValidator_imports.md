# Implementation: SecurityValidator Import Methods

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Security]], [[domain::Static_Analysis]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Concrete AST visitor methods for detecting and validating import statements against security allowlists.

=== Description ===

These methods are part of the `SecurityValidator` class (extends `ast.NodeVisitor`):

1. **`visit_Import()`**: Handles `import X` and `import X as Y` statements
   - Iterates through all imported names
   - Calls `_validate_import()` for each module

2. **`visit_ImportFrom()`**: Handles `from X import Y` statements
   - Checks for relative imports (level > 0) which are always blocked
   - Validates the source module against allowlists

Both methods track already-checked modules in `self.checked_modules` to avoid duplicate validation.

=== Usage ===

These methods are called automatically by `ast.NodeVisitor.visit()` during AST traversal. They populate `self.violations` list with any detected import violations.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/task-runner-python/src/task_analyzer.py
* '''Lines:''' L34-50

=== Signature ===
<syntaxhighlight lang="python">
class SecurityValidator(ast.NodeVisitor):
    """AST visitor that enforces import allowlists and blocks dangerous attribute access."""

    def __init__(self, security_config: SecurityConfig):
        self.checked_modules: set[str] = set()
        self.violations: list[str] = []
        self.security_config = security_config

    def visit_Import(self, node: ast.Import) -> None:
        """
        Detect bare import statements (e.g., import os),
        including aliased (e.g., import numpy as np).
        """

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """
        Detect from import statements (e.g., from os import path).
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from src.task_analyzer import SecurityValidator
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| node || ast.Import or ast.ImportFrom || Yes || AST node from traversal
|}

=== Outputs ===
{| class="wikitable"
|-
! Side Effect !! Description
|-
| self.violations || Violation strings appended if import not allowed
|-
| self.checked_modules || Module names added to prevent duplicate checks
|}

== Usage Examples ==

=== Import Statement Detection ===
<syntaxhighlight lang="python">
# Code being analyzed
code = """
import os
import numpy as np
from json import loads, dumps
from os.path import join
from . import sibling
"""

# AST nodes generated:
# 1. Import(names=[alias(name='os')])
#    → visit_Import called, validates 'os'
#
# 2. Import(names=[alias(name='numpy', asname='np')])
#    → visit_Import called, validates 'numpy'
#
# 3. ImportFrom(module='json', names=[alias(name='loads'), alias(name='dumps')])
#    → visit_ImportFrom called, validates 'json'
#
# 4. ImportFrom(module='os.path', names=[alias(name='join')])
#    → visit_ImportFrom called, validates 'os.path' → extracts 'os'
#
# 5. ImportFrom(level=1, module=None, names=[alias(name='sibling')])
#    → visit_ImportFrom called, level > 0 → relative import violation
</syntaxhighlight>

=== Validation Flow ===
<syntaxhighlight lang="python">
def visit_Import(self, node: ast.Import) -> None:
    for alias in node.names:
        module_name = alias.name
        self._validate_import(module_name, node.lineno)
    self.generic_visit(node)

def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
    if node.level > 0:
        # Relative import: from . import X
        self._add_violation(node.lineno, ERROR_RELATIVE_IMPORT)
    elif node.module:
        # Absolute import: from X import Y
        self._validate_import(node.module, node.lineno)

    self.generic_visit(node)

def _validate_import(self, module_path: str, lineno: int) -> None:
    # Extract root module: "os.path" → "os"
    module_name = module_path.split(".")[0]

    # Skip if already checked
    if module_name in self.checked_modules:
        return

    self.checked_modules.add(module_name)

    # Validate against allowlists
    is_allowed, error_msg = validate_module_import(
        module_path, self.security_config
    )

    if not is_allowed:
        self._add_violation(lineno, error_msg)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Import_Analysis]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Task_Runner_Env]]
