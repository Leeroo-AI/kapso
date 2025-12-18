# Principle: Security Validation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
* [[source::Doc|Python AST|https://docs.python.org/3/library/ast.html]]
|-
! Domains
| [[domain::Security]], [[domain::Static_Analysis]], [[domain::Code_Execution]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Principle for pre-execution static analysis of Python code to detect security violations before running untrusted code in a sandbox.

=== Description ===

Security Validation is a critical first line of defense in sandboxed code execution. It uses Abstract Syntax Tree (AST) analysis to detect:

1. **Disallowed Imports**: Modules not in the allowlist (stdlib or external)
2. **Dangerous Names**: Access to `__builtins__`, `__globals__`, `__loader__`, etc.
3. **Dangerous Attributes**: Access to `__code__`, `__subclasses__`, `func_globals`, etc.
4. **Dynamic Imports**: Use of `__import__()` with non-constant arguments
5. **Relative Imports**: Import statements using relative paths

This static analysis happens **before** code execution, enabling fast rejection of obviously malicious code without the overhead of runtime enforcement.

=== Usage ===

Apply this principle when:
- Executing user-provided Python code in any context
- Building code evaluation or REPL systems
- Implementing plugin architectures that load untrusted code
- Creating educational platforms that run student code

== Theoretical Basis ==

Security validation uses **AST Visitor Pattern** for static analysis:

<syntaxhighlight lang="python">
# Pseudo-code for security validation

def validate(code: str) -> None:
    # 1. Check cache for previous validation
    cache_key = (hash(code), allowlists)
    if cached_result := cache.get(cache_key):
        if cached_result.has_violations:
            raise SecurityViolationError(cached_result.violations)
        return  # Previously validated safe

    # 2. Parse code into AST
    tree = ast.parse(code)

    # 3. Walk AST looking for violations
    validator = SecurityValidator(security_config)
    validator.visit(tree)

    # 4. Cache result and raise if violations found
    cache[cache_key] = validator.violations
    if validator.violations:
        raise SecurityViolationError(violations)
</syntaxhighlight>

Key detection patterns:
- **Import Statements**: `visit_Import`, `visit_ImportFrom`
- **Name Access**: `visit_Name` for blocked identifiers
- **Attribute Access**: `visit_Attribute` for dangerous dunder methods
- **Function Calls**: `visit_Call` for dynamic import detection
- **Subscript Access**: `visit_Subscript` for `__builtins__["key"]` patterns

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_TaskAnalyzer_validate]]
