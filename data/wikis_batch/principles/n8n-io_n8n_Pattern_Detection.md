{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|n8n Python Task Runner|https://docs.n8n.io]]
|-
! Domains
| [[domain::Security]], [[domain::Static_Analysis]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Detecting dangerous code patterns through systematic traversal of abstract syntax trees to identify security violations.

=== Description ===

Pattern detection is a static analysis technique that identifies security-sensitive code constructs by matching AST node patterns against known dangerous operations. The detector traverses the entire syntax tree and checks each node against patterns that represent security violations, such as accessing reflection attributes, manipulating code objects, or accessing blocked namespaces.

This principle solves the problem of detecting sophisticated attacks:
* Direct blocking of dangerous functions is insufficient
* Attackers use attribute access to bypass name checks
* Reflection capabilities enable sandbox escapes
* Code object manipulation can inject arbitrary code

Pattern detection catches these attacks by analyzing the structure of the code rather than just the names being used.

=== Usage ===

Apply this principle when:
* Implementing security validators for code execution
* Building static analysis security testing tools
* Detecting anti-patterns or code smells
* Enforcing coding standards programmatically

== Theoretical Basis ==

Pattern detection uses AST visitor pattern with security rules:

<pre>
Dangerous Patterns to Detect:

1. Reflection Attribute Access:
   obj.__code__      # Access to function bytecode
   obj.__globals__   # Access to global namespace
   obj.__builtins__  # Access to builtin functions
   obj.__dict__      # Access to object internals

2. Code Object Manipulation:
   func.func_code    # Old-style code object access
   frame.f_locals    # Stack frame manipulation
   frame.f_globals   # Global namespace via frame

3. Blocked Names:
   eval(...)         # Dynamic code execution
   exec(...)         # Statement execution
   compile(...)      # Code compilation
   __import__(...)   # Dynamic imports

Detection Algorithm:
  class SecurityValidator(ast.NodeVisitor):
    DANGEROUS_ATTRS = {"__code__", "__globals__", "__builtins__", ...}
    BLOCKED_NAMES = {"eval", "exec", "compile", ...}

    def visit_Attribute(self, node):
      # Check attribute access patterns
      if node.attr in DANGEROUS_ATTRS:
        record_violation(f"Access to {node.attr}")

      self.generic_visit(node)

    def visit_Name(self, node):
      # Check name references
      if node.id in BLOCKED_NAMES:
        record_violation(f"Use of {node.id}")

      self.generic_visit(node)

Pattern Matching:
  Attribute(value=Name(id="func"), attr="__code__")
    -> VIOLATION: Accessing function code object

  Name(id="eval")
    -> VIOLATION: Using eval function

  Call(func=Name(id="getattr"), args=[_, Constant(value="__globals__")])
    -> VIOLATION: Using getattr to access __globals__
</pre>

The pattern detection approach is comprehensive because it analyzes the code structure rather than relying on string matching or simple name checks that can be easily bypassed.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_SecurityValidator_visit_Attribute]]

=== Related Principles ===
* [[related::Principle:n8n-io_n8n_AST_Parsing]]
* [[related::Principle:n8n-io_n8n_Violation_Reporting]]
