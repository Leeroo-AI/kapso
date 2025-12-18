# Environment: Sandbox Execution Environment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n-io/n8n|https://github.com/n8n-io/n8n]]
* [[source::Doc|task_executor.py|packages/@n8n/task-runner-python/src/task_executor.py]]
|-
! Domains
| [[domain::Security]], [[domain::Sandboxing]], [[domain::Runtime_Isolation]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==
Isolated subprocess environment with filtered builtins, sanitized sys.modules, and runtime import validation for secure Python code execution.

=== Description ===
This environment represents the **restricted runtime context** inside the forked subprocess where user Python code actually executes. It is not a separate installation requirement but rather a dynamically constructed sandbox within the Python Task Runner. The sandbox is created by:
1. Clearing `os.environ` (if `runner_env_deny` is enabled)
2. Sanitizing `sys.modules` to remove disallowed modules
3. Filtering `__builtins__` to remove dangerous functions
4. Wrapping `__import__` with security validation

=== Usage ===
This environment is automatically created by `TaskExecutor._all_items()` and `TaskExecutor._per_item()` before executing user code. It enforces the security policies defined in `SecurityConfig` (stdlib_allow, external_allow, builtins_deny).

== System Requirements ==

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux or macOS || Inherits from parent Python Task Runner environment
|-
| Python || >= 3.13 || Same as parent environment
|-
| Process || Forked subprocess || Created via `forkserver` multiprocessing context
|}

== Dependencies ==

This is a **virtual environment** - it does not have its own dependencies. Instead, it **restricts** access to the parent environment's modules based on allowlists.

=== Available by Default ===
The following are always available in the sandbox (required for error handling):
* `builtins` - Filtered Python builtins
* `__main__` - Main module context
* `sys` - System module (for stdlib detection)
* `traceback` - For error stack traces
* `linecache` - For traceback line reading
* `importlib` - For import machinery
* `importlib.machinery` - For import hooks

=== Conditionally Available ===
Additional modules are available based on `SecurityConfig`:
* Modules in `stdlib_allow` set
* Modules in `external_allow` set
* Use `*` in either allowlist to permit all modules of that category

== Credentials ==

The sandbox environment by default **blocks access to all environment variables**. This is controlled by `N8N_BLOCK_RUNNER_ENV_ACCESS` (default: `true`).

When `runner_env_deny` is `true`:
* `os.environ` is cleared before code execution
* User code cannot access secrets or credentials from the host

== Code Evidence ==

Environment clearing from `task_executor.py:195-196`:
<syntaxhighlight lang="python">
if security_config.runner_env_deny:
    os.environ.clear()
</syntaxhighlight>

Module sanitization from `task_executor.py:442-477`:
<syntaxhighlight lang="python">
@staticmethod
def _sanitize_sys_modules(security_config: SecurityConfig):
    safe_modules = {
        "builtins",
        "__main__",
        "sys",
        "traceback",
        "linecache",
        "importlib",
        "importlib.machinery",
    }

    if "*" in security_config.stdlib_allow:
        safe_modules.update(sys.stdlib_module_names)
    else:
        safe_modules.update(security_config.stdlib_allow)

    if "*" in security_config.external_allow:
        safe_modules.update(
            name
            for name in sys.modules.keys()
            if name not in sys.stdlib_module_names
        )
    else:
        safe_modules.update(security_config.external_allow)

    # keep modules marked as safe and submodules of those
    safe_prefixes = [safe + "." for safe in safe_modules]
    modules_to_remove = [
        name
        for name in sys.modules.keys()
        if name not in safe_modules
        and not any(name.startswith(prefix) for prefix in safe_prefixes)
    ]

    for module_name in modules_to_remove:
        del sys.modules[module_name]
</syntaxhighlight>

Builtin filtering from `task_executor.py:424-439`:
<syntaxhighlight lang="python">
@staticmethod
def _filter_builtins(security_config: SecurityConfig):
    """Get __builtins__ with denied ones removed."""

    if len(security_config.builtins_deny) == 0:
        filtered = dict(__builtins__)
    else:
        filtered = {
            k: v
            for k, v in __builtins__.items()
            if k not in security_config.builtins_deny
        }

    filtered["__import__"] = TaskExecutor._create_safe_import(security_config)

    return filtered
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `Security violation detected` + `Import of standard library module 'X' is disallowed` || User code imports a disallowed stdlib module || Add module to `N8N_RUNNERS_STDLIB_ALLOW` or use `*`
|-
|| `Security violation detected` + `Import of external package 'X' is disallowed` || User code imports a disallowed external package || Add package to `N8N_RUNNERS_EXTERNAL_ALLOW` or use `*`
|-
|| `Access to attribute 'X' is disallowed` || User code accesses a blocked attribute (e.g., `__globals__`) || Remove usage of blocked attribute from code
|-
|| `Access to name 'X' is disallowed` || User code accesses a blocked name (e.g., `__builtins__`) || Remove usage of blocked name from code
|-
|| `Dynamic __import__() calls are not allowed` || User code uses `__import__()` with non-constant argument || Use standard import statements with static names
|}

== Compatibility Notes ==

* **Blocked Builtins (Default):** `eval`, `exec`, `compile`, `open`, `input`, `breakpoint`, `getattr`, `object`, `type`, `vars`, `setattr`, `delattr`, `hasattr`, `dir`, `memoryview`, `__build_class__`, `globals`, `locals`, `license`, `help`, `credits`, `copyright`
* **Blocked Names:** `__loader__`, `__builtins__`, `__globals__`, `__spec__`, `__name__`
* **Blocked Attributes:** 35+ dangerous attributes including `__subclasses__`, `__globals__`, `f_code`, `f_back`, `__class__`, etc.
* **Relative Imports:** Always disallowed
* **Name Mangling:** Access to `_ClassName__attr` patterns is blocked

== Related Pages ==

* [[requires_env::Implementation:n8n-io_n8n_TaskExecutor_execute]]
* [[requires_env::Implementation:n8n-io_n8n_TaskExecutor_sandbox]]
* [[requires_env::Implementation:n8n-io_n8n_TaskExecutor_put_result]]
* [[requires_env::Implementation:n8n-io_n8n_validate_module_import]]
