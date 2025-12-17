{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Security]], [[domain::Code_Analysis]], [[domain::Import_Control]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Concrete tool for validating Python import statements against security allowlists, detecting both direct imports and from-imports during AST traversal.

=== Description ===

The `SecurityValidator` class extends Python's `ast.NodeVisitor` to implement two visitor methods: `visit_Import()` and `visit_ImportFrom()`. These methods are automatically invoked when the AST walker encounters import statements, enabling validation against the configured import allowlist and detection of relative imports.

=== Usage ===

This implementation is used during the static analysis phase of task validation. After parsing code with `ast.parse()`, the `SecurityValidator` is invoked to traverse the AST. When import nodes are encountered, these methods validate each imported module against the security configuration's allowlist, raising violations for unauthorized imports or relative imports.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/task-runner-python/src/task_analyzer.py
* '''Lines:''' L34-50

=== Signature ===
<syntaxhighlight lang="python">
def visit_Import(self, node: ast.Import) -> None:
    """Detect bare import statements."""

def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
    """Detect from import statements."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from src.task_analyzer import SecurityValidator
from src.security_config import SecurityConfig
import ast
</syntaxhighlight>

== I/O Contract ==

=== Inputs (visit_Import) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| node || ast.Import || Yes || AST node representing an import statement
|}

=== Inputs (visit_ImportFrom) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| node || ast.ImportFrom || Yes || AST node representing a from-import statement
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| (none) || None || Methods update internal violations list as side effect
|}

=== Side Effects ===
* Appends violation strings to `self.violations` list for disallowed imports
* Adds checked modules to `self.checked_modules` set
* Calls `self.generic_visit(node)` to continue AST traversal

== Implementation Details ==

=== visit_Import Implementation ===
<syntaxhighlight lang="python">
def visit_Import(self, node: ast.Import) -> None:
    """Detect bare import statements"""
    for alias in node.names:
        self._validate_import(alias.name, node.lineno)
    self.generic_visit(node)
</syntaxhighlight>

This method handles statements like:
<syntaxhighlight lang="python">
import os
import sys, json
import pandas as pd
</syntaxhighlight>

Key behaviors:
* Iterates through all imported modules in the statement (handles comma-separated imports)
* Extracts module name from `alias.name` (e.g., "pandas" even if aliased as "pd")
* Passes module name and line number to `_validate_import()`
* Continues traversal with `generic_visit()` to check nested structures

=== visit_ImportFrom Implementation ===
<syntaxhighlight lang="python">
def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
    """Detect from import statements"""
    if node.level > 0:
        self._add_violation(node.lineno, ERROR_RELATIVE_IMPORT)
    elif node.module:
        self._validate_import(node.module, node.lineno)
</syntaxhighlight>

This method handles statements like:
<syntaxhighlight lang="python">
from os import path
from ..parent import module  # Relative import
from package.submodule import function
</syntaxhighlight>

Key behaviors:
* '''Relative import detection''': `node.level > 0` indicates relative imports (dots before module name)
* '''Absolute import validation''': If `node.module` exists, validates against allowlist
* '''Special cases''': Handles `from . import x` where module is None

=== Import Validation Logic ===
The `_validate_import()` helper (called by both methods) performs:

<syntaxhighlight lang="python">
def _validate_import(self, module_name: str, lineno: int) -> None:
    """Check if module is allowed by security configuration."""
    # Extract top-level module (e.g., "os.path" -> "os")
    top_level_module = module_name.split('.')[0]

    # Skip if already checked
    if top_level_module in self.checked_modules:
        return

    self.checked_modules.add(top_level_module)

    # Validate against allowlist
    is_allowed, error_msg = validate_module_import(
        top_level_module,
        self.security_config
    )

    if not is_allowed:
        self._add_violation(lineno, error_msg)
</syntaxhighlight>

=== Error Messages ===
The implementation uses predefined error constants:
* '''ERROR_RELATIVE_IMPORT''': "Relative imports are not allowed for security reasons"
* '''ERROR_UNAUTHORIZED_IMPORT''': "Module '{module}' is not in the allowlist"

== Usage Examples ==

=== Basic Import Validation ===
<syntaxhighlight lang="python">
import ast
from src.task_analyzer import SecurityValidator
from src.security_config import SecurityConfig

# Configure allowed modules
security_config = SecurityConfig()
security_config.import_allowlist = ["pandas", "numpy", "json"]

# Code to validate
code = """
import pandas as pd
import numpy as np
import os  # Not in allowlist
"""

# Parse and validate
tree = ast.parse(code)
validator = SecurityValidator(security_config)
validator.visit(tree)

# Check violations
if validator.violations:
    print("Security violations found:")
    for violation in validator.violations:
        print(f"  {violation}")
    # Output:
    # Line 3: Module 'os' is not in the allowlist
</syntaxhighlight>

=== Relative Import Detection ===
<syntaxhighlight lang="python">
import ast
from src.task_analyzer import SecurityValidator

# Code with relative imports
code = """
from . import sibling_module
from ..parent import utils
from ...grandparent import config
"""

tree = ast.parse(code)
validator = SecurityValidator(security_config)
validator.visit(tree)

print(f"Violations: {len(validator.violations)}")  # 3
# All relative imports are blocked
</syntaxhighlight>

=== From-Import Validation ===
<syntaxhighlight lang="python">
import ast
from src.task_analyzer import SecurityValidator

security_config = SecurityConfig()
security_config.import_allowlist = ["os", "json"]

code = """
from os import path, environ
from json import loads, dumps
from subprocess import run  # Not allowed
"""

tree = ast.parse(code)
validator = SecurityValidator(security_config)
validator.visit(tree)

# Only subprocess violation reported
assert len(validator.violations) == 1
assert "subprocess" in validator.violations[0]
</syntaxhighlight>

=== Submodule Import Handling ===
<syntaxhighlight lang="python">
import ast
from src.task_analyzer import SecurityValidator

security_config = SecurityConfig()
security_config.import_allowlist = ["pandas"]

code = """
import pandas.core.frame
from pandas.io import json
"""

tree = ast.parse(code)
validator = SecurityValidator(security_config)
validator.visit(tree)

# No violations - "pandas" is allowed, submodules inherit permission
assert len(validator.violations) == 0
</syntaxhighlight>

=== Multiple Imports in One Statement ===
<syntaxhighlight lang="python">
import ast
from src.task_analyzer import SecurityValidator

security_config = SecurityConfig()
security_config.import_allowlist = ["json", "sys"]

code = """
import json, sys, os  # Mixed: 2 allowed, 1 not
"""

tree = ast.parse(code)
validator = SecurityValidator(security_config)
validator.visit(tree)

# Only 'os' violation reported
assert len(validator.violations) == 1
assert "os" in validator.violations[0]
</syntaxhighlight>

=== Deduplication via checked_modules ===
<syntaxhighlight lang="python">
import ast
from src.task_analyzer import SecurityValidator

security_config = SecurityConfig()
security_config.import_allowlist = []

code = """
import os
import os.path
from os import environ
"""

tree = ast.parse(code)
validator = SecurityValidator(security_config)
validator.visit(tree)

# Only one violation despite three import statements
# All check the same top-level module 'os'
assert len(validator.violations) == 1
</syntaxhighlight>

== Security Rationale ==

=== Why Block Relative Imports ===
Relative imports are blocked because:
* They depend on the task execution directory structure
* They could access parent directories outside the task sandbox
* They make security auditing more difficult (module path not explicit)

=== Why Validate Top-Level Module Only ===
The implementation validates `module.split('.')[0]` because:
* Submodules inherit the security properties of their parent
* Prevents allowlist explosion (don't need "os.path", "os.environ", etc.)
* Consistent with Python's import system (importing parent imports all submodules)

=== Why Track checked_modules ===
The `checked_modules` set prevents duplicate violations:
* Same module may be imported multiple times with different syntax
* Reduces noise in violation reports
* Improves performance by skipping redundant validation

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Import_Validation]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Task_Runner]]

=== Related Implementations ===
* [[related::Implementation:n8n-io_n8n_ast_parse]]
* [[related::Implementation:n8n-io_n8n_SecurityValidator_visit_Attribute]]
* [[related::Implementation:n8n-io_n8n_TaskExecutor_create_safe_import]]

=== Used By Workflow ===
* [[used_by::Workflow:n8n-io_n8n_Security_Validation]]
