{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|n8n Python Task Runner|https://docs.n8n.io]]
|-
! Domains
| [[domain::Security]], [[domain::Sandboxing]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Restricting the set of built-in functions available to executed code by removing dangerous capabilities from the runtime environment.

=== Description ===

Builtin filtering is a runtime security control that modifies the Python execution environment before running user code by removing dangerous built-in functions from the __builtins__ namespace. This prevents user code from accessing capabilities like dynamic code execution (eval, exec, compile), import mechanisms (__import__), or other sensitive operations, even if they bypass static analysis.

This principle solves critical runtime security problems:
* Static analysis cannot catch dynamically constructed function names
* Reflection and getattr can bypass name checks
* Code may use obfuscation to hide dangerous operations
* Defense-in-depth requires runtime controls in addition to static validation

By removing dangerous builtins before execution, the system ensures that even sophisticated attacks cannot access forbidden capabilities.

=== Usage ===

Apply this principle when:
* Executing untrusted code in sandboxed environments
* Implementing interpreter-level security controls
* Building secure code evaluation platforms
* Enforcing principle of least privilege at runtime

== Theoretical Basis ==

Builtin filtering implements privilege restriction through environment modification:

<pre>
Dangerous Builtins to Remove:
  eval      - Evaluates string as Python expression
  exec      - Executes string as Python statements
  compile   - Compiles source into code object
  __import__- Dynamic module importing
  open      - File I/O operations
  input     - User input (can be abused)
  globals   - Access to global namespace
  locals    - Access to local namespace
  vars      - Access to object attributes

Filtering Algorithm:
  function create_safe_builtins():
    # Start with standard builtins
    safe_builtins = dict(__builtins__)

    # Remove dangerous functions
    for dangerous in BLOCKED_BUILTINS:
      safe_builtins.pop(dangerous, None)

    # Replace __import__ with safe version
    safe_builtins['__import__'] = create_safe_import()

    return safe_builtins

Execution with Filtered Builtins:
  safe_builtins = create_safe_builtins()
  safe_globals = {
    '__builtins__': safe_builtins,
    'items': input_data,
    ...
  }

  exec(user_code, safe_globals)

Attack Mitigation:
  # These attacks are blocked by builtin filtering:

  getattr(__builtins__, 'eval')('malicious')
    -> KeyError: 'eval' not in __builtins__

  __import__('os').system('ls')
    -> Caught by safe_import wrapper

  [x for x in [].__class__.__bases__[0].__subclasses__()]
    -> Can't reach dangerous classes without builtins

Security Properties:
  - Removes dangerous capabilities entirely
  - Cannot be bypassed via getattr/reflection
  - Complements static analysis
  - Enforces least privilege
</pre>

Builtin filtering provides defense-in-depth by ensuring that runtime execution cannot access dangerous capabilities even if static analysis is bypassed through obfuscation or dynamic code construction.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_TaskExecutor_filter_builtins]]

=== Related Principles ===
* [[related::Principle:n8n-io_n8n_Runtime_Import_Validation]]
* [[related::Principle:n8n-io_n8n_Pattern_Detection]]
