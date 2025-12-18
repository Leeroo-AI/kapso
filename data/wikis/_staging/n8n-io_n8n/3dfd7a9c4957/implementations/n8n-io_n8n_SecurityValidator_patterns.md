# Implementation: SecurityValidator Pattern Detection Methods

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

Concrete AST visitor methods for detecting dangerous code patterns that could bypass sandbox restrictions.

=== Description ===

These methods detect various sandbox escape patterns:

1. **`visit_Name()`**: Detects direct access to blocked names like `__builtins__`
2. **`visit_Attribute()`**: Detects attribute access to blocked attributes like `__class__`
3. **`visit_Call()`**: Detects calls to `__import__()` with dynamic arguments
4. **`visit_Subscript()`**: Detects dictionary access like `__builtins__["key"]`

All methods add violations to `self.violations` with line numbers for error reporting.

=== Usage ===

These methods are called automatically during AST traversal. They work together to detect various forms of the same dangerous patterns (e.g., `__builtins__` vs `__builtins__["key"]`).

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/task-runner-python/src/task_analyzer.py
* '''Lines:''' L52-129

=== Signature ===
<syntaxhighlight lang="python">
class SecurityValidator(ast.NodeVisitor):

    def visit_Name(self, node: ast.Name) -> None:
        """Detect access to blocked names like __builtins__."""

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Detect access to unsafe attributes that could bypass security."""

    def visit_Call(self, node: ast.Call) -> None:
        """Detect calls to __import__() that could bypass security."""

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Detect dict access to blocked attributes, e.g. __builtins__['__spec__']"""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from src.task_analyzer import SecurityValidator
from src.constants import BLOCKED_NAMES, BLOCKED_ATTRIBUTES
</syntaxhighlight>

== I/O Contract ==

=== Blocked Names (BLOCKED_NAMES) ===
{| class="wikitable"
|-
! Name !! Reason
|-
| __loader__ || Access to import system internals
|-
| __builtins__ || Access to builtin functions
|-
| __globals__ || Access to global namespace
|-
| __spec__ || Module specification object
|-
| __name__ || Module name (can be used for reflection)
|}

=== Blocked Attributes (BLOCKED_ATTRIBUTES - 35+ entries) ===
{| class="wikitable"
|-
! Attribute !! Reason
|-
| __class__ || Type introspection for escape
|-
| __bases__ || Parent class access
|-
| __subclasses__ || Find all subclasses for exploitation
|-
| __mro__ || Method resolution order traversal
|-
| __code__ || Code object manipulation
|-
| __globals__, func_globals || Global namespace access
|-
| __reduce__, __reduce_ex__ || Pickle-based attacks
|-
| __getattribute__ || Attribute access bypass
|-
| __import__ || Dynamic import access
|-
| ... || (35+ more patterns)
|}

== Usage Examples ==

=== Name Detection ===
<syntaxhighlight lang="python">
def visit_Name(self, node: ast.Name) -> None:
    if node.id in BLOCKED_NAMES:
        self._add_violation(
            node.lineno,
            ERROR_DANGEROUS_NAME.format(name=node.id)
        )
    self.generic_visit(node)

# Detects:
# x = __builtins__  → "Access to dangerous name '__builtins__' is not allowed"
</syntaxhighlight>

=== Attribute Detection ===
<syntaxhighlight lang="python">
def visit_Attribute(self, node: ast.Attribute) -> None:
    # Check blocked attributes
    if node.attr in BLOCKED_ATTRIBUTES:
        self._add_violation(
            node.lineno,
            ERROR_DANGEROUS_ATTRIBUTE.format(attr=node.attr)
        )

    # Check name-mangled attributes (_ClassName__attr)
    if node.attr.startswith("_") and "__" in node.attr:
        parts = node.attr.split("__", 1)
        if len(parts) == 2 and parts[0].startswith("_"):
            self._add_violation(node.lineno, ERROR_NAME_MANGLED_ATTRIBUTE)

    self.generic_visit(node)

# Detects:
# ().__class__.__bases__[0].__subclasses__()
# → Multiple violations for __class__, __bases__, __subclasses__
</syntaxhighlight>

=== Dynamic Import Detection ===
<syntaxhighlight lang="python">
def visit_Call(self, node: ast.Call) -> None:
    is_import_call = (
        (isinstance(node.func, ast.Name) and node.func.id == "__import__")
        or (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "__import__"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in {"builtins", "__builtins__"}
        )
    )

    if is_import_call:
        if node.args and isinstance(node.args[0], ast.Constant):
            # Constant string - validate the module
            module_name = node.args[0].value
            self._validate_import(module_name, node.lineno)
        else:
            # Dynamic argument - always blocked
            self._add_violation(node.lineno, ERROR_DYNAMIC_IMPORT)

# Detects:
# __import__("os")  → Validates "os" against allowlists
# __import__(user_input)  → "Dynamic import is not allowed"
</syntaxhighlight>

=== Subscript Access Detection ===
<syntaxhighlight lang="python">
def visit_Subscript(self, node: ast.Subscript) -> None:
    is_builtins_access = (
        (isinstance(node.value, ast.Name)
         and node.value.id in {"__builtins__", "builtins"})
        or (isinstance(node.value, ast.Attribute)
            and node.value.attr in {"__builtins__", "builtins"})
    )

    if (is_builtins_access
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)):
        key = node.slice.value
        if key in BLOCKED_ATTRIBUTES:
            self._add_violation(
                node.lineno,
                ERROR_DANGEROUS_ATTRIBUTE.format(attr=key)
            )

# Detects:
# __builtins__["__import__"]  → Blocked attribute access
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Dangerous_Pattern_Detection]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Task_Runner_Env]]
