# Principle: Runtime Import Validation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Security]], [[domain::Runtime_Enforcement]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Principle for validating module imports at runtime to enforce security policies for dynamic imports that cannot be detected by static analysis.

=== Description ===

Runtime Import Validation is the second line of defense after static analysis:

1. **Module Classification**: Determines if module is stdlib or external using `sys.stdlib_module_names`
2. **Allowlist Checking**: Validates against appropriate allowlist (stdlib_allow or external_allow)
3. **Wildcard Support**: `"*"` in allowlist permits all modules of that type
4. **Error Messages**: Generates descriptive errors listing allowed modules

This runtime validation catches:
- Imports from `eval()` or `exec()` (if not blocked)
- Imports via `importlib.import_module()` (if not blocked)
- Any import that bypassed static analysis

=== Usage ===

Apply this principle when:
- Implementing defense-in-depth for import security
- Building systems where static analysis may be incomplete
- Creating configurable import restrictions
- Designing sandboxes that need runtime enforcement

== Theoretical Basis ==

Runtime validation follows a **Policy Gate** pattern:

<syntaxhighlight lang="python">
# Pseudo-code for runtime import validation

def validate_module_import(module_path: str, config: SecurityConfig) -> tuple[bool, str | None]:
    # Extract root module: "os.path" → "os"
    module_name = module_path.split(".")[0]

    # Classify module type
    is_stdlib = module_name in sys.stdlib_module_names
    is_external = not is_stdlib

    # Check against appropriate allowlist
    if is_stdlib:
        if "*" in config.stdlib_allow or module_name in config.stdlib_allow:
            return (True, None)  # Allowed
        else:
            error = f"Import '{module_path}' is not allowed. Allowed stdlib: {config.stdlib_allow}"
            return (False, error)

    if is_external:
        if "*" in config.external_allow or module_name in config.external_allow:
            return (True, None)  # Allowed
        else:
            error = f"Import '{module_path}' is not allowed. Allowed external: {config.external_allow}"
            return (False, error)
</syntaxhighlight>

Key: Used by both static analysis (TaskAnalyzer) and runtime (safe_import wrapper).

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_validate_module_import]]
