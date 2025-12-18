# Principle: AST Parsing

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Python AST|https://docs.python.org/3/library/ast.html]]
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Static_Analysis]], [[domain::Security]], [[domain::Parsing]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Principle for converting Python source code into an Abstract Syntax Tree representation that can be traversed for security analysis.

=== Description ===

AST Parsing is the foundation of static security analysis:

1. **Lexical Analysis**: Python tokenizes source code into tokens
2. **Syntax Analysis**: Tokens are parsed into a hierarchical tree structure
3. **AST Nodes**: Each language construct becomes a typed node (Import, Name, Call, etc.)
4. **Error Detection**: Syntax errors are caught during parsing, before execution

The AST provides:
- **Structural Access**: Navigate code by construct type, not text patterns
- **Complete Coverage**: Every code path is represented in the tree
- **Type Information**: Nodes carry semantic information (import vs function call)
- **Position Tracking**: Line numbers for error reporting

=== Usage ===

Apply this principle when:
- Implementing static analysis tools
- Building linters or code quality checkers
- Creating security validators that need to understand code structure
- Designing code transformation tools

== Theoretical Basis ==

AST parsing converts linear source code into a tree structure:

<syntaxhighlight lang="python">
# Source code
source = """
import os
x = os.path.join("a", "b")
"""

# AST representation (simplified)
Module(body=[
    Import(names=[alias(name='os')]),
    Assign(
        targets=[Name(id='x')],
        value=Call(
            func=Attribute(
                value=Attribute(
                    value=Name(id='os'),
                    attr='path'
                ),
                attr='join'
            ),
            args=[Constant(value='a'), Constant(value='b')]
        )
    )
])
</syntaxhighlight>

The `ast.NodeVisitor` pattern allows selective processing:

<syntaxhighlight lang="python">
class SecurityValidator(ast.NodeVisitor):
    def visit_Import(self, node):
        # Process import statements
        for alias in node.names:
            self.check_module(alias.name)
        self.generic_visit(node)  # Visit children

    def visit_Name(self, node):
        # Process name references
        if node.id in BLOCKED_NAMES:
            self.violations.append(f"Line {node.lineno}: ...")
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_ast_parse]]
