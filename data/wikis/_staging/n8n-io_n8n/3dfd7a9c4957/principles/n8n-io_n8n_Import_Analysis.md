# Principle: Import Analysis

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
* [[source::Doc|Python AST|https://docs.python.org/3/library/ast.html]]
|-
! Domains
| [[domain::Security]], [[domain::Static_Analysis]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Principle for detecting and validating import statements in Python code against security allowlists using AST visitor methods.

=== Description ===

Import Analysis detects all forms of Python imports in user code:

1. **Bare Imports**: `import os` → `visit_Import` captures module name
2. **From Imports**: `from os import path` → `visit_ImportFrom` captures module
3. **Aliased Imports**: `import numpy as np` → alias doesn't affect validation
4. **Relative Imports**: `from . import module` → Always rejected (level > 0)

Each detected import is validated against:
- **stdlib_allow**: Set of permitted standard library modules
- **external_allow**: Set of permitted third-party packages

Violations are collected with line numbers for error reporting.

=== Usage ===

Apply this principle when:
- Building sandboxed code execution environments
- Implementing module import restrictions
- Creating security policies for code evaluation
- Designing plugin systems with limited capabilities

== Theoretical Basis ==

Import analysis uses AST visitor methods:

<syntaxhighlight lang="python">
# Pseudo-code for import analysis

class SecurityValidator(ast.NodeVisitor):

    def visit_Import(self, node: ast.Import):
        """
        Handle: import os
                import numpy as np
        """
        for alias in node.names:
            module_name = alias.name  # 'os' or 'numpy'
            self._validate_import(module_name, node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """
        Handle: from os import path
                from json import loads, dumps
                from . import sibling  (relative - always blocked)
        """
        if node.level > 0:  # Relative import
            self._add_violation(node.lineno, "Relative imports not allowed")
        elif node.module:
            self._validate_import(node.module, node.lineno)
        self.generic_visit(node)

    def _validate_import(self, module_path: str, lineno: int):
        module_name = module_path.split(".")[0]  # 'os.path' → 'os'

        is_allowed, error = validate_module_import(module_path, config)
        if not is_allowed:
            self._add_violation(lineno, error)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_SecurityValidator_imports]]
