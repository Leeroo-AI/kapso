# Principle: Dangerous Pattern Detection

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

Principle for detecting dangerous Python patterns that could bypass security restrictions, including dunder attribute access, name-mangled attributes, and dynamic imports.

=== Description ===

Dangerous Pattern Detection identifies code patterns that could escape the sandbox:

1. **Blocked Names**: Direct access to `__builtins__`, `__globals__`, `__loader__`, `__spec__`, `__name__`
2. **Blocked Attributes**: Access to attributes like `__class__`, `__bases__`, `__subclasses__`, `__code__`, `func_globals`, etc.
3. **Name-Mangled Attributes**: Attributes starting with `_ClassName__attr` pattern
4. **Dynamic Imports**: Calls to `__import__()` with non-constant arguments
5. **Subscript Access**: `__builtins__["__spec__"]` style dictionary access to blocked keys

These patterns are commonly used in sandbox escape exploits to:
- Access parent classes and find vulnerable subclasses
- Retrieve function code objects and modify them
- Access global namespaces to escape restriction
- Dynamically import blocked modules

=== Usage ===

Apply this principle when:
- Building Python sandboxes with untrusted code
- Implementing security validators for code execution
- Creating CTF challenges or sandbox environments
- Designing defense against Python jail escapes

== Theoretical Basis ==

Pattern detection uses multiple AST visitor methods:

<syntaxhighlight lang="python">
# Blocked patterns

# 1. Blocked Names - direct identifier access
BLOCKED_NAMES = {"__builtins__", "__globals__", "__loader__", "__spec__", "__name__"}
x = __builtins__  # Blocked by visit_Name

# 2. Blocked Attributes - attribute access
BLOCKED_ATTRIBUTES = {
    "__class__", "__bases__", "__subclasses__", "__mro__",
    "__code__", "__globals__", "func_globals",
    "__reduce__", "__reduce_ex__", ...
}
x.__class__.__bases__  # Each blocked by visit_Attribute

# 3. Name-Mangled Attributes
class Foo:
    __secret = 1
obj._Foo__secret  # Blocked: starts with _ and contains __

# 4. Dynamic Import
__import__(user_input)  # Blocked: non-constant argument to __import__

# 5. Subscript Access to builtins
__builtins__["__import__"]  # Blocked by visit_Subscript
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_SecurityValidator_patterns]]
