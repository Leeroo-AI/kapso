{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|n8n Python Task Runner|https://docs.n8n.io]]
|-
! Domains
| [[domain::Security]], [[domain::Code_Analysis]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Validating that code only imports approved modules from a predefined allowlist to prevent malicious code execution.

=== Description ===

Import validation is a security control that restricts which Python modules can be imported by user-provided code. The validator checks all import statements (both "import X" and "from X import Y" forms) against allowlists of approved modules. Any import not on the allowlist is rejected as a security violation.

This principle solves critical security problems:
* Prevents access to dangerous system modules (os, subprocess, sys)
* Blocks network access modules (socket, urllib, requests)
* Restricts file system access (io, pathlib, shutil)
* Prevents code execution modules (eval, exec via importlib)
* Blocks relative imports that could access runner internals

The allowlist model provides defense-in-depth by limiting the attack surface even if other security controls are bypassed.

=== Usage ===

Apply this principle when:
* Executing untrusted user-provided code
* Implementing sandboxed execution environments
* Building secure code evaluation platforms
* Restricting capabilities of plugin or extension systems

== Theoretical Basis ==

Import validation implements allowlist-based security:

<pre>
Allowlist Model:
  ALLOWED = {
    "stdlib": {"json", "math", "datetime", "re", ...},
    "external": {"numpy", "pandas", "requests", ...}
  }

  DEFAULT_POLICY = DENY_ALL

Validation Algorithm:
  function validate_import(module_name, from_parts):
    # Block relative imports
    if module_name.startswith('.'):
      raise SecurityViolation("Relative imports not allowed")

    # Get top-level module
    top_level = module_name.split('.')[0]

    # Check against allowlists
    if top_level in ALLOWED.stdlib:
      return ALLOW
    if top_level in ALLOWED.external:
      return ALLOW

    return DENY

AST Patterns to Check:
  Import(names=[alias(name="os")])
    -> Check "os" against allowlist

  ImportFrom(module="os.path", names=[alias(name="join")])
    -> Check "os" against allowlist

  ImportFrom(module=None, names=[...], level=1)
    -> Reject (relative import)

Security Properties:
  - Default deny (closed world)
  - Explicit allowlist required
  - Transitive imports also restricted
  - No runtime bypass possible
</pre>

The allowlist approach is more secure than blocklist because it requires explicit approval for each capability rather than trying to enumerate all dangerous operations.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_SecurityValidator_visit_Import]]

=== Related Principles ===
* [[related::Principle:n8n-io_n8n_Runtime_Import_Validation]]
* [[related::Principle:n8n-io_n8n_AST_Parsing]]
