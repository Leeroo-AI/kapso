{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
* [[source::Doc|n8n Docs|https://docs.n8n.io]]
|-
! Domains
| [[domain::Security]], [[domain::Static_Analysis]], [[domain::Sandboxing]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

Multi-layered security validation process for Python code execution in n8n, combining static AST analysis with runtime import validation and builtin filtering.

=== Description ===

This workflow describes the defense-in-depth security architecture of the Python Task Runner. Security validation occurs at two layers: static analysis before execution (parsing code as AST to detect dangerous patterns), and runtime validation during execution (intercepting imports and filtering builtins). This dual approach catches both explicit violations in the source code and dynamic attempts to bypass restrictions.

=== Usage ===

This workflow is automatically invoked as part of every Python task execution. It cannot be bypassed. Use this documentation when:
* Understanding why certain Python code fails with SecurityViolationError
* Configuring allowlists for stdlib and external modules
* Auditing the security posture of the task runner
* Extending security checks for new threat vectors

== Execution Steps ==

=== Step 1: Code Reception ===
[[step::Principle:n8n-io_n8n_Task_Settings_Reception]]

When task settings arrive from the broker, the code string is extracted along with execution context (node_mode, items, query). The code represents user-authored Python that will run in the workflow context.

'''Input validation:'''
* Code is a string field from the BrokerTaskSettings message
* Empty code will fail at AST parsing stage
* Code is not modified before analysis

=== Step 2: Cache Lookup ===
[[step::Principle:n8n-io_n8n_Validation_Caching]]

Before parsing, the analyzer checks a validation cache keyed by (code_hash, allowlists_tuple). Cache hits return immediately (either passing or re-raising cached violations). This optimization prevents repeated expensive AST parsing of identical code.

'''Cache behavior:'''
* Uses SHA-256 hash of code string
* Allowlists included in cache key (different configs = different entries)
* LRU eviction at 500 entries
* Cache shared across all TaskAnalyzer instances

=== Step 3: AST Parsing ===
[[step::Principle:n8n-io_n8n_AST_Parsing]]

The code is parsed into an Abstract Syntax Tree using Python's ast.parse(). Syntax errors at this stage terminate validation with a SyntaxError (distinct from SecurityViolationError).

'''Parsing notes:'''
* Full Python 3 syntax supported
* Parse mode is "exec" (module-level code)
* Error messages include line/column information

=== Step 4: Import Statement Validation ===
[[step::Principle:n8n-io_n8n_Import_Validation]]

The SecurityValidator AST visitor traverses the tree, checking all import forms against configured allowlists. Both stdlib and external modules have separate allowlists with "*" meaning "allow all".

'''Import patterns checked:'''
* `import module` - bare imports
* `import module as alias` - aliased imports
* `from module import name` - from imports
* `from .relative import name` - relative imports (always blocked)
* `__import__('module')` - dynamic imports with literal arguments
* `__import__(variable)` - dynamic imports with non-literal args (always blocked)

=== Step 5: Dangerous Pattern Detection ===
[[step::Principle:n8n-io_n8n_Pattern_Detection]]

Beyond imports, the validator detects attribute access patterns that could bypass security restrictions. These include introspection attributes that expose code internals and name-mangled attributes used for class internals.

'''Blocked patterns:'''
* Dangerous attributes: __code__, __globals__, __builtins__, __spec__, etc.
* Name-mangled attributes: patterns like _Class__private
* Blocked names configured in constants (e.g., breakpoint)
* Subscript access to builtins dict (e.g., __builtins__['__spec__'])

=== Step 6: Violation Aggregation ===
[[step::Principle:n8n-io_n8n_Violation_Reporting]]

All violations found during AST traversal are collected with line numbers. If any violations exist, a SecurityViolationError is raised with a multi-line description listing all issues.

'''Error format:'''
* Each violation: "Line N: description"
* All violations reported together (not fail-fast)
* Cached for future lookups of same code

=== Step 7: Runtime Builtin Filtering ===
[[step::Principle:n8n-io_n8n_Builtin_Filtering]]

At subprocess creation time, the __builtins__ dict is filtered to remove denied builtins (e.g., eval, exec). A safe import wrapper replaces __import__ to enforce allowlists at runtime.

'''Runtime protections:'''
* Builtins filtered based on builtins_deny config
* __import__ replaced with validating wrapper
* sys.modules sanitized to remove unsafe cached modules
* Environment variables optionally cleared

=== Step 8: Runtime Import Interception ===
[[step::Principle:n8n-io_n8n_Runtime_Import_Validation]]

The safe import wrapper validates each import at runtime against the same allowlists used in static analysis. This catches dynamically constructed imports that static analysis couldn't verify.

'''Runtime validation:'''
* Validates module root (e.g., os.path checks "os")
* Checks stdlib vs external classification
* Returns original import result if allowed
* Raises SecurityViolationError if denied

== Execution Diagram ==

{{#mermaid:graph TD
    A[Code Received] --> B{Cache Hit?}
    B -->|Yes, Clean| C[Return Success]
    B -->|Yes, Violations| D[Raise Cached Error]
    B -->|No| E[Parse AST]
    E -->|SyntaxError| F[Return SyntaxError]
    E -->|Success| G[Visit Import Nodes]
    G --> H[Visit Attribute Nodes]
    H --> I[Visit Call Nodes]
    I --> J[Visit Subscript Nodes]
    J --> K{Any Violations?}
    K -->|No| L[Cache & Pass]
    K -->|Yes| M[Cache & Raise SecurityViolationError]
    L --> N[Filter Builtins]
    N --> O[Install Safe Import]
    O --> P[Sanitize sys.modules]
    P --> Q[Execute with Runtime Checks]
}}

== Related Pages ==

* [[step::Principle:n8n-io_n8n_Task_Settings_Reception]]
* [[step::Principle:n8n-io_n8n_Validation_Caching]]
* [[step::Principle:n8n-io_n8n_AST_Parsing]]
* [[step::Principle:n8n-io_n8n_Import_Validation]]
* [[step::Principle:n8n-io_n8n_Pattern_Detection]]
* [[step::Principle:n8n-io_n8n_Violation_Reporting]]
* [[step::Principle:n8n-io_n8n_Builtin_Filtering]]
* [[step::Principle:n8n-io_n8n_Runtime_Import_Validation]]
