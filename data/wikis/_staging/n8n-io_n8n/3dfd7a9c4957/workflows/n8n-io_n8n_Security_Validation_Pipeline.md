# Security Validation Pipeline

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
* [[source::Doc|Python AST|https://docs.python.org/3/library/ast.html]]
|-
! Domains
| [[domain::Security]], [[domain::Static_Analysis]], [[domain::Python]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:30 GMT]]
|}

== Overview ==

Multi-layered security validation process for Python code execution, combining AST-based static analysis with runtime import validation and sandbox enforcement.

=== Description ===

This workflow documents n8n's defense-in-depth approach to secure Python code execution:

1. **Goal:** Prevent malicious code from escaping the sandbox, accessing sensitive resources, or bypassing security restrictions.
2. **Scope:** Covers pre-execution static analysis, runtime import validation, builtin filtering, and sys.modules sanitization.
3. **Strategy:** Defense in depth with multiple independent security layers - if one layer is bypassed, others provide protection.

The security model uses allowlists (not blocklists) for imports, ensuring only explicitly permitted modules can be used. Runtime hooks intercept import calls and validate them against the same policies enforced at AST analysis time.

=== Usage ===

Execute this workflow when:
- Configuring security policies for the Python task runner
- Understanding how code is validated before execution
- Investigating security-related task failures
- Auditing the security model for potential vulnerabilities
- Adding new allowed modules to the security configuration

== Execution Steps ==

=== Step 1: Configuration Loading ===
[[step::Principle:n8n-io_n8n_Security_Configuration]]

Load security configuration from environment variables to define the validation policies. The configuration specifies which standard library modules, external packages, and builtins are allowed or denied.

'''Key considerations:'''
* N8N_RUNNERS_STDLIB_ALLOW: Comma-separated list of allowed stdlib modules (or "*" for all)
* N8N_RUNNERS_EXTERNAL_ALLOW: Comma-separated list of allowed external packages (or "*" for all)
* N8N_RUNNERS_BUILTINS_DENY: Comma-separated list of blocked builtins
* N8N_BLOCK_RUNNER_ENV_ACCESS: Whether to clear environment variables in subprocess
* Default builtins_deny includes: eval, exec, compile, open, input, getattr, setattr, etc.

=== Step 2: Cache Lookup ===
[[step::Principle:n8n-io_n8n_Validation_Caching]]

Check the validation cache for previously analyzed code to avoid redundant AST parsing. The cache uses a composite key of code hash and allowlist configuration to ensure cached results are invalidated when policies change.

'''Pseudocode:'''
  1. Compute SHA-256 hash of code string
  2. Create cache key: (code_hash, sorted_allowlists_tuple)
  3. Check OrderedDict cache (LRU with max 500 entries)
  4. If hit: move to end (LRU update), return cached violations
  5. If miss: proceed to AST analysis

=== Step 3: AST Parsing ===
[[step::Principle:n8n-io_n8n_AST_Parsing]]

Parse the Python code into an Abstract Syntax Tree for static analysis. The AST provides a structured representation of the code that can be traversed to identify security-relevant patterns.

'''What happens:'''
* Python's ast.parse() converts code string to AST
* SyntaxError raised for invalid Python code
* AST nodes represent imports, names, attributes, calls, subscripts
* SecurityValidator extends ast.NodeVisitor for traversal

=== Step 4: Import Analysis ===
[[step::Principle:n8n-io_n8n_Import_Analysis]]

Traverse the AST to identify all import statements and validate them against the allowlist policies. This catches disallowed imports at analysis time before any code executes.

'''Key considerations:'''
* Detects `import X` and `from X import Y` statements
* Validates against stdlib_allow and external_allow lists
* Tracks checked modules to avoid duplicate validation
* Relative imports (from . import) are always blocked
* Dynamic __import__() calls require constant string arguments

=== Step 5: Dangerous Pattern Detection ===
[[step::Principle:n8n-io_n8n_Dangerous_Pattern_Detection]]

Scan the AST for access to dangerous names, attributes, and patterns that could be used to bypass security restrictions. This provides defense against introspection-based attacks.

'''Blocked patterns:'''
* Names: __loader__, __builtins__, __globals__, __spec__, __name__
* Attributes: __subclasses__, __code__, __globals__, f_back, f_globals, etc.
* Name-mangled attributes: _ClassName__attr pattern
* Builtin dict subscript: __builtins__['__spec__']
* Dynamic __import__ calls with non-constant arguments

=== Step 6: Violation Aggregation ===
[[step::Principle:n8n-io_n8n_Violation_Aggregation]]

Collect all security violations found during AST traversal and format them for error reporting. Violations include line numbers and descriptive messages for user debugging.

'''Output format:'''
* List of strings: "Line N: description"
* Multiple violations aggregated into single error
* Cache populated with violations list (even if empty)
* SecurityViolationError raised if violations exist

=== Step 7: Runtime Import Validation ===
[[step::Principle:n8n-io_n8n_Runtime_Import_Validation]]

Install a custom __import__ hook that validates every import at runtime, providing a second layer of defense. This catches dynamic imports that may not be detectable via static analysis.

'''Pseudocode:'''
  1. Replace __import__ in filtered builtins
  2. On import: extract base module name
  3. Check against stdlib_allow (or sys.stdlib_module_names if "*")
  4. Check against external_allow
  5. Raise SecurityViolationError if not allowed
  6. Delegate to original __import__ if allowed

=== Step 8: Sandbox Environment Setup ===
[[step::Principle:n8n-io_n8n_Sandbox_Environment]]

Configure the execution sandbox with filtered builtins, sanitized sys.modules, and restricted environment variables. This establishes the security boundaries for code execution.

'''What happens:'''
* Clear os.environ if runner_env_deny is set
* Filter __builtins__ to remove denied functions
* Remove non-allowed modules from sys.modules
* Keep safe prefixes (submodules of allowed modules)
* Install validating __import__ in filtered builtins

== Execution Diagram ==

{{#mermaid:graph TD
    A[Configuration Loading] --> B[Cache Lookup]
    B -->|Cache Hit| C{Violations?}
    B -->|Cache Miss| D[AST Parsing]
    D --> E[Import Analysis]
    E --> F[Dangerous Pattern Detection]
    F --> G[Violation Aggregation]
    G --> H[Update Cache]
    H --> C
    C -->|Yes| I[Raise SecurityViolationError]
    C -->|No| J[Runtime Import Validation]
    J --> K[Sandbox Environment Setup]
    K --> L[Code Execution]
}}

== Related Pages ==

=== Steps ===
* [[step::Principle:n8n-io_n8n_Security_Configuration]]
* [[step::Principle:n8n-io_n8n_Validation_Caching]]
* [[step::Principle:n8n-io_n8n_AST_Parsing]]
* [[step::Principle:n8n-io_n8n_Import_Analysis]]
* [[step::Principle:n8n-io_n8n_Dangerous_Pattern_Detection]]
* [[step::Principle:n8n-io_n8n_Violation_Aggregation]]
* [[step::Principle:n8n-io_n8n_Runtime_Import_Validation]]
* [[step::Principle:n8n-io_n8n_Sandbox_Environment]]

=== Related Concepts ===
* [[related::Workflow:n8n-io_n8n_Python_Task_Execution]] - Parent workflow using this validation
