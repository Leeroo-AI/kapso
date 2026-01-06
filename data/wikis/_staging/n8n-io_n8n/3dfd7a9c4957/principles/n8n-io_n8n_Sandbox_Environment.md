# Principle: Sandbox Environment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Security]], [[domain::Sandboxing]], [[domain::Runtime_Enforcement]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Principle for constructing a restricted Python runtime environment by filtering builtins, sanitizing module state, and wrapping imports with security checks.

=== Description ===

Sandbox Environment creates the restricted execution context:

1. **Builtin Filtering**: Creates copy of `__builtins__` with denied functions removed
2. **Import Wrapping**: Replaces `__import__` with validation wrapper
3. **Module Sanitization**: Removes non-allowlisted modules from `sys.modules`
4. **Environment Clearing**: Optionally clears `os.environ` to prevent leakage

This multi-layered approach ensures:
- Denied builtins (eval, exec, compile, open) are unavailable
- Every import goes through validation
- Pre-loaded modules can't be accessed to bypass restrictions
- Environment variables don't leak sensitive data

=== Usage ===

Apply this principle when:
- Creating sandboxed code execution environments
- Building multi-tenant systems with untrusted code
- Implementing plugin architectures with security boundaries
- Designing educational platforms that run student code

== Theoretical Basis ==

Sandbox construction follows a **Constrained Environment** pattern:

<syntaxhighlight lang="python">
# Pseudo-code for sandbox environment setup

def setup_sandbox(security_config):
    # 1. Filter builtins - remove denied functions
    filtered_builtins = {
        k: v for k, v in __builtins__.items()
        if k not in security_config.builtins_deny
    }

    # 2. Replace __import__ with safe wrapper
    filtered_builtins["__import__"] = create_safe_import(security_config)

    # 3. Sanitize sys.modules - remove non-allowed modules
    safe_modules = {"builtins", "sys", ...} | allowlists
    for name in list(sys.modules.keys()):
        if name not in safe_modules and not starts_with_safe_prefix(name):
            del sys.modules[name]

    return filtered_builtins

def create_safe_import(security_config):
    original_import = __builtins__["__import__"]

    def safe_import(name, *args, **kwargs):
        is_allowed, error = validate_module_import(name, security_config)
        if not is_allowed:
            raise SecurityViolationError(description=error)
        return original_import(name, *args, **kwargs)

    return safe_import
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_TaskExecutor_sandbox]]
