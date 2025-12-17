{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|n8n Python Task Runner|https://docs.n8n.io]]
|-
! Domains
| [[domain::Security]], [[domain::Runtime_Validation]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Validating import statements at runtime to catch dynamic imports that bypass static analysis and enforce security policies during execution.

=== Description ===

Runtime import validation is a defense-in-depth security control that intercepts all import operations during code execution and validates them against the same security allowlists used in static analysis. The system replaces Python's built-in __import__ function with a wrapper that checks each import request before allowing it to proceed.

This principle solves the limitation of static analysis:
* Dynamic imports using variables cannot be analyzed statically
* Obfuscated import statements may evade AST pattern matching
* Indirect imports through allowed modules need validation
* Runtime provides the final enforcement point for security policy

Runtime validation acts as a second layer of defense, ensuring that even sophisticated evasion techniques cannot bypass import restrictions.

=== Usage ===

Apply this principle when:
* Implementing defense-in-depth security strategies
* Executing code that may use dynamic imports
* Protecting against obfuscated malicious code
* Enforcing security policies at multiple layers

== Theoretical Basis ==

Runtime import validation implements the interception pattern:

<pre>
Import Interception:
  Original Flow:
    user_code -> __import__('module') -> Python loader -> module loaded

  Intercepted Flow:
    user_code -> safe_import_wrapper -> validate -> __import__ -> module loaded
                                            |
                                            v
                                    raise SecurityError (if invalid)

Safe Import Implementation:
  function create_safe_import(allowed_stdlib, allowed_external):
    original_import = __builtins__.__import__

    function safe_import(name, globals, locals, fromlist, level):
      # Block relative imports
      if level != 0:
        raise SecurityError("Relative imports not allowed")

      # Get top-level module
      top_level = name.split('.')[0]

      # Validate against allowlists
      if top_level not in allowed_stdlib and top_level not in allowed_external:
        raise SecurityError(f"Import of '{name}' is not allowed")

      # Delegate to original import
      return original_import(name, globals, locals, fromlist, level)

    return safe_import

Dynamic Import Scenarios Caught:
  # Static analysis cannot detect these:

  module_name = "o" + "s"
  __import__(module_name)  # Runtime validation catches this

  getattr(__builtins__, '__import__')('subprocess')
    # Caught by safe_import wrapper

  importlib.import_module(user_input)
    # importlib uses __import__ internally, caught by wrapper

Defense-in-Depth Properties:
  Layer 1 (Static): AST analysis catches obvious violations
  Layer 2 (Runtime): Import wrapper catches dynamic violations

  Both layers enforce the same allowlist policy
  Runtime provides final enforcement point
  No bypass possible without compromising Python interpreter

Performance Considerations:
  - Import interception has minimal overhead
  - Validation is simple hash table lookup
  - Only happens once per module (imports are cached)
  - Static analysis remains primary validator (catches errors early)
</pre>

Runtime import validation complements static analysis by providing enforcement at the actual point of import execution, ensuring that security policies cannot be bypassed through dynamic code construction or obfuscation techniques.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_TaskExecutor_create_safe_import]]

=== Related Principles ===
* [[related::Principle:n8n-io_n8n_Import_Validation]]
* [[related::Principle:n8n-io_n8n_Builtin_Filtering]]
* [[related::Principle:n8n-io_n8n_AST_Parsing]]
