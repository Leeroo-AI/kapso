{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Documentation|Python AST|https://docs.python.org/3/library/ast.html]]
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Security]], [[domain::Code_Analysis]], [[domain::AST]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

External tool from Python's standard library for parsing Python source code into an Abstract Syntax Tree (AST), enabling static code analysis without execution.

=== Description ===

`ast.parse()` is a built-in Python function that converts Python source code (as a string) into an AST representation. The n8n task runner uses this to analyze task code for security violations before execution, enabling detection of dangerous imports, attribute access, and other patterns.

=== Usage ===

This implementation is invoked by the `TaskAnalyzer` when validating Python code submitted for task execution. The AST is then traversed by the `SecurityValidator` to detect security policy violations such as unauthorized imports, access to blocked attributes, or use of dangerous names.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/task-runner-python/src/task_analyzer.py
* '''Lines:''' L188 (usage context)
* '''Documentation:''' [https://docs.python.org/3/library/ast.html Python AST Module]

=== Signature ===
<syntaxhighlight lang="python">
def parse(
    source: str | bytes,
    filename: str = '<unknown>',
    mode: str = 'exec',
    *,
    type_comments: bool = False,
    feature_version: int | tuple[int, int] | None = None
) -> ast.Module | ast.Expression | ast.Interactive:
    """Parse Python source code into an AST."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
import ast

# Or specific components
from ast import parse, NodeVisitor, Import, ImportFrom, Attribute, Name
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| source || str or bytes || Yes || Python source code to parse
|-
| filename || str || No || Filename for error messages (default: '&lt;unknown&gt;')
|-
| mode || str || No || Compilation mode: 'exec' (statements), 'eval' (expression), 'single' (interactive)
|-
| type_comments || bool || No || Enable PEP 484 type comment parsing (default: False)
|-
| feature_version || int or tuple || No || Python version to emulate for parsing (default: current version)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| ast_node || ast.Module or ast.Expression or ast.Interactive || Root node of the parsed AST
|}

=== Exceptions ===
{| class="wikitable"
|-
! Exception !! Condition
|-
| SyntaxError || Raised when source code contains invalid Python syntax
|-
| ValueError || Raised when mode is not 'exec', 'eval', or 'single'
|}

== Implementation Details ==

=== AST Structure ===
The returned AST is a tree of node objects representing the syntactic structure:
* '''Module nodes''': Represent complete Python programs (mode='exec')
* '''Expression nodes''': Represent single expressions (mode='eval')
* '''Statement nodes''': Individual statements (if, while, import, etc.)
* '''Expression nodes''': Values, operators, function calls, etc.

Each node has:
* '''Node type''': Import, Attribute, Call, BinOp, etc.
* '''Fields''': Type-specific data (e.g., Import has 'names' field)
* '''Location info''': lineno, col_offset for error reporting

=== AST Visitor Pattern ===
The AST module provides the `NodeVisitor` base class for traversing ASTs:
<syntaxhighlight lang="python">
class MyVisitor(ast.NodeVisitor):
    def visit_Import(self, node):
        print(f"Found import at line {node.lineno}")
        self.generic_visit(node)

tree = ast.parse(code)
visitor = MyVisitor()
visitor.visit(tree)
</syntaxhighlight>

The n8n `SecurityValidator` extends `NodeVisitor` to implement custom security checks.

=== Performance Characteristics ===
* '''Time Complexity''': O(n) where n is code length
* '''Space Complexity''': O(m) where m is number of AST nodes
* '''Caching''': n8n caches validation results to avoid repeated parsing

=== Mode Selection ===
* '''exec mode''' (n8n usage): Parses complete programs with statements
* '''eval mode''': Parses single expressions only
* '''single mode''': Parses single interactive statements

== Usage Examples ==

=== Basic Parsing (n8n Context) ===
<syntaxhighlight lang="python">
import ast
from src.task_analyzer import SecurityValidator

# Code submitted for task execution
task_code = """
import pandas as pd
import numpy as np

def process_data(df):
    return df.describe()
"""

# Parse into AST
try:
    tree = ast.parse(task_code)
    print(f"Parsed successfully: {type(tree)}")  # <class 'ast.Module'>

    # Perform security validation
    validator = SecurityValidator(security_config)
    validator.visit(tree)

    if validator.violations:
        print(f"Security violations found: {validator.violations}")
    else:
        print("Code passed security validation")

except SyntaxError as e:
    print(f"Syntax error at line {e.lineno}: {e.msg}")
</syntaxhighlight>

=== Examining AST Structure ===
<syntaxhighlight lang="python">
import ast

code = """
import os
x = os.path.join('a', 'b')
"""

tree = ast.parse(code)

# Dump AST structure for debugging
print(ast.dump(tree, indent=2))

# Output shows:
# Module(
#   body=[
#     Import(names=[alias(name='os')]),
#     Assign(
#       targets=[Name(id='x')],
#       value=Call(
#         func=Attribute(
#           value=Attribute(value=Name(id='os'), attr='path'),
#           attr='join'
#         ),
#         args=[Constant(value='a'), Constant(value='b')]
#       )
#     )
#   ]
# )
</syntaxhighlight>

=== Error Handling ===
<syntaxhighlight lang="python">
import ast

# Invalid Python code
invalid_code = """
import pandas
def broken_function(
    # Missing closing parenthesis
"""

try:
    tree = ast.parse(invalid_code)
except SyntaxError as e:
    print(f"Syntax error detected:")
    print(f"  Line {e.lineno}: {e.msg}")
    print(f"  Text: {e.text}")
    print(f"  Offset: {' ' * (e.offset - 1)}^")

    # In n8n context, this would reject the task before execution
    raise SecurityViolationError(
        message="Code contains syntax errors",
        description=f"Line {e.lineno}: {e.msg}"
    )
</syntaxhighlight>

=== Security Validation Pattern ===
<syntaxhighlight lang="python">
import ast

class ImportDetector(ast.NodeVisitor):
    def __init__(self):
        self.imports = []

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append((alias.name, node.lineno))
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.append((node.module, node.lineno))
        self.generic_visit(node)

code = """
import os
import sys
from pathlib import Path
"""

tree = ast.parse(code)
detector = ImportDetector()
detector.visit(tree)

print("Imports found:")
for module, line in detector.imports:
    print(f"  Line {line}: {module}")
</syntaxhighlight>

== Security Considerations ==

=== Safe by Design ===
`ast.parse()` is completely safe as it only parses code without executing it. This is why n8n uses AST analysis as the first security layer - violations can be detected before any code runs.

=== Limitations ===
AST analysis cannot detect:
* '''Runtime imports''': `__import__()` called with dynamic strings
* '''Eval/exec usage''': Dynamic code execution
* '''Obfuscated code''': Code that hides intent through complexity

These require runtime protection via `_create_safe_import()` and `_filter_builtins()`.

=== Common Patterns ===
AST analysis excels at detecting:
* Import statements (static)
* Attribute access patterns
* Function/method calls
* Name usage
* Dangerous syntax patterns

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_AST_Parsing]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Task_Runner]]

=== Related Implementations ===
* [[related::Implementation:n8n-io_n8n_SecurityValidator_visit_Import]]
* [[related::Implementation:n8n-io_n8n_SecurityValidator_visit_Attribute]]
* [[related::Implementation:n8n-io_n8n_TaskAnalyzer_validate]]

=== Used By Workflow ===
* [[used_by::Workflow:n8n-io_n8n_Security_Validation]]

=== External Documentation ===
* [https://docs.python.org/3/library/ast.html Python AST Module Documentation]
* [https://greentreesnakes.readthedocs.io/en/latest/ Green Tree Snakes - AST Tutorial]
