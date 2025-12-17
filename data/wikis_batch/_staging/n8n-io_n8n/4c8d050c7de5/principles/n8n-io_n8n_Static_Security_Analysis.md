{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|n8n Python Task Runner|https://docs.n8n.io]]
|-
! Domains
| [[domain::Task_Execution]], [[domain::Security]], [[domain::Code_Analysis]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Pre-execution static analysis of Python code using Abstract Syntax Tree (AST) parsing to detect and prevent security violations before runtime.

=== Description ===

Static security analysis examines user-submitted Python code at compile time to identify dangerous patterns, disallowed imports, and security violations before the code ever executes. By parsing code into an AST and walking the tree structure, the analyzer can detect:
* Prohibited module imports (filesystem access, network operations, subprocess spawning)
* Dangerous function calls (eval, exec, compile, __import__)
* Restricted language features (async/await, decorators, class definitions)
* Suspicious patterns (bytecode manipulation, reflection abuse)

This approach implements defense-in-depth security: rather than relying solely on runtime sandboxing, violations are caught early in the compilation phase. Early detection provides better error messages, reduces attack surface, and prevents resource consumption from malicious code.

The analysis is deterministic and can be performed quickly without executing code, making it suitable for high-throughput task systems where security and performance are both critical.

=== Usage ===

Apply this principle when building systems that execute untrusted code where:
* Security requirements prohibit certain language features or libraries
* Runtime sandboxing alone is insufficient (defense-in-depth needed)
* Early rejection of invalid code improves user experience
* Performance cost of AST analysis is acceptable vs execution cost
* Clear policy about allowed/disallowed operations can be defined
* System needs to prevent resource consumption from malicious code
* Audit trail of security violations is required

== Theoretical Basis ==

Static analysis operates on the **Abstract Syntax Tree**, a structured representation of source code:

**Analysis Pipeline:**
```
Source Code → Parse → AST → Walk Tree → Validate Nodes → Accept/Reject
```

**Key Detection Techniques:**

1. **Import Analysis**:
   ```python
   # Detect: import os, subprocess, socket
   ast.Import nodes: Check module names against blocklist
   ast.ImportFrom nodes: Check module and imported names
   ```

2. **Function Call Analysis**:
   ```python
   # Detect: eval(), exec(), __import__()
   ast.Call nodes: Check func.id or func.attr against dangerous functions
   ```

3. **Attribute Access Analysis**:
   ```python
   # Detect: __code__, __globals__, __builtins__
   ast.Attribute nodes: Check attr names for reflection/introspection
   ```

4. **Language Feature Detection**:
   ```python
   # Detect: async def, @decorator, class definitions
   ast.AsyncFunctionDef, ast.ClassDef: Reject based on policy
   ```

**Security Properties:**

* **Soundness**: All violations are detected (no false negatives for defined rules)
* **Completeness**: Only actual violations are rejected (minimize false positives)
* **Determinism**: Same code always produces same analysis result
* **Performance**: O(n) in AST nodes, independent of code complexity

**Example Validation Logic:**
```python
def validate_import(node: ast.Import):
    BLOCKED_MODULES = {'os', 'sys', 'subprocess', 'socket'}
    for alias in node.names:
        if alias.name in BLOCKED_MODULES:
            raise SecurityViolation(f"Import {alias.name} not allowed")
```

**Limitations:**
* Cannot detect runtime-constructed imports: `__import__('o' + 's')`
* Cannot analyze dynamically generated code
* May have false positives on safe but complex patterns
* Complementary to runtime sandboxing, not a replacement

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_TaskAnalyzer_validate]]

=== Related Principles ===
* [[Principle:n8n-io_n8n_Subprocess_Isolation]] - Runtime security layer
* [[Principle:n8n-io_n8n_Code_Execution]] - Executes code after static validation
* [[Principle:n8n-io_n8n_Task_Acceptance]] - Validation occurs during acceptance phase
