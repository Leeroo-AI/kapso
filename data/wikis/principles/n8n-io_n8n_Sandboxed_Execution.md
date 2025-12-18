# Principle: Sandboxed Execution

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
* [[source::Doc|Python exec|https://docs.python.org/3/library/functions.html#exec]]
|-
! Domains
| [[domain::Security]], [[domain::Code_Execution]], [[domain::Sandboxing]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Principle for executing untrusted Python code within a constrained environment with restricted builtins, controlled imports, and sanitized system state.

=== Description ===

Sandboxed Execution is the runtime security layer that complements static analysis:

1. **Environment Clearing**: Optionally clears `os.environ` to prevent env variable leakage
2. **Module Sanitization**: Removes non-allowlisted modules from `sys.modules`
3. **Builtin Filtering**: Creates restricted `__builtins__` dict with denied functions removed
4. **Import Wrapping**: Replaces `__import__` with a validation wrapper
5. **Code Wrapping**: Wraps user code in a function to control output capture
6. **Controlled Globals**: Provides minimal globals with input data and custom print

This defense-in-depth approach ensures that even if static analysis misses something, runtime controls prevent exploitation.

=== Usage ===

Apply this principle when:
- Running any code from untrusted sources
- Building notebook or REPL environments
- Implementing plugin systems with user-provided code
- Creating educational platforms that execute student code

== Theoretical Basis ==

Sandboxed execution follows a **Constrained Environment** pattern:

<syntaxhighlight lang="python">
# Pseudo-code for sandboxed execution

def execute_sandboxed(code, items, security_config):
    # 1. Clear environment if configured
    if security_config.runner_env_deny:
        os.environ.clear()

    # 2. Sanitize sys.modules
    safe_modules = {"builtins", "sys", ...} | allowlists
    for module in list(sys.modules.keys()):
        if module not in safe_modules:
            del sys.modules[module]

    # 3. Filter builtins
    filtered_builtins = {
        k: v for k, v in __builtins__.items()
        if k not in denied_builtins
    }
    filtered_builtins["__import__"] = safe_import_wrapper

    # 4. Wrap user code
    wrapped = f"""
def _user_function():
{indent(code, '    ')}

_output = _user_function()
"""

    # 5. Execute with minimal globals
    globals = {
        "__builtins__": filtered_builtins,
        "_items": items,
        "print": custom_print,
    }
    exec(compile(wrapped), globals)

    return globals["_output"]
</syntaxhighlight>

Key security layers:
- **Process Isolation**: Subprocess boundary
- **Module Control**: sys.modules sanitization
- **Builtin Restriction**: Filtered __builtins__
- **Import Validation**: Runtime import checks

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_TaskExecutor_execute]]
