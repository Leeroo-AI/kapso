# Implementation: ast.parse (Wrapper Doc)

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Python AST|https://docs.python.org/3/library/ast.html]]
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Static_Analysis]], [[domain::Parsing]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Python standard library function for parsing source code into an Abstract Syntax Tree, used in the security validation pipeline.

=== Description ===

`ast.parse()` is the Python stdlib function that converts source code strings into AST tree structures. In the n8n task runner, it's used to:

1. Parse user-submitted Python code before execution
2. Generate an AST tree for security validation traversal
3. Detect syntax errors early (before sandbox execution)

The function is called within `TaskAnalyzer.validate()` after cache miss.

=== External Reference ===

* [https://docs.python.org/3/library/ast.html#ast.parse Official Python Documentation: ast.parse]

=== n8n-Specific Usage ===

In the task runner context:
- Called with just the code string (default mode="exec")
- Result is passed to `SecurityValidator.visit(tree)`
- SyntaxError propagates to caller, failing the task

== Code Reference ==

=== Source Location ===
* '''Library:''' Python Standard Library
* '''n8n Usage:''' packages/@n8n/task-runner-python/src/task_analyzer.py:L188

=== Signature ===
<syntaxhighlight lang="python">
def parse(
    source: str | bytes,
    filename: str = "<unknown>",
    mode: str = "exec",
    *,
    type_comments: bool = False,
    feature_version: tuple[int, int] | None = None
) -> ast.Module:
    """
    Parse source into an AST node.

    Args:
        source: Python source code as string or bytes.
        filename: Name to use in error messages.
        mode: "exec" for module, "eval" for expression, "single" for interactive.
        type_comments: If True, parse PEP 484 type comments.
        feature_version: Target Python version tuple.

    Returns:
        ast.Module node representing the parsed code.

    Raises:
        SyntaxError: If source code has invalid syntax.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
import ast

# Usage in task_analyzer.py
tree = ast.parse(code)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| source || str | bytes || Yes || Python source code to parse
|-
| filename || str || No || Filename for error messages (default: "<unknown>")
|-
| mode || str || No || Parse mode: "exec", "eval", or "single" (default: "exec")
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| (returns) || ast.Module || Root AST node containing the parsed code structure
|-
| (raises) || SyntaxError || If source code is syntactically invalid
|}

== Usage Examples ==

=== Basic Parsing in TaskAnalyzer ===
<syntaxhighlight lang="python">
# From task_analyzer.py:L188
def validate(self, code: str) -> None:
    # ... cache lookup ...

    # Parse code into AST
    tree = ast.parse(code)

    # Traverse for security violations
    security_validator = SecurityValidator(self._security_config)
    security_validator.visit(tree)

    # ... cache result ...
</syntaxhighlight>

=== AST Node Types ===
<syntaxhighlight lang="python">
import ast

code = """
import os
from json import loads
x = __builtins__
y = obj.__class__.__bases__
"""

tree = ast.parse(code)

# tree.body contains:
# [0] Import(names=[alias(name='os')])
# [1] ImportFrom(module='json', names=[alias(name='loads')])
# [2] Assign(targets=[Name(id='x')], value=Name(id='__builtins__'))
# [3] Assign(targets=[Name(id='y')], value=Attribute(...))
</syntaxhighlight>

=== Error Handling ===
<syntaxhighlight lang="python">
# Syntax errors propagate to TaskRunner._execute_task
try:
    tree = ast.parse("def broken(")
except SyntaxError as e:
    # e.lineno, e.offset, e.text available
    # Task fails with this error
    pass
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_AST_Parsing]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Task_Runner_Env]]
