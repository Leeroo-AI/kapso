# Implementation: TaskExecutor Sandbox Methods

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Security]], [[domain::Sandboxing]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Concrete static methods for constructing the sandboxed execution environment by filtering builtins and sanitizing module state.

=== Description ===

These methods prepare the subprocess environment for safe code execution:

1. **`_filter_builtins()`**: Creates a restricted `__builtins__` dictionary
   - Copies original `__builtins__` excluding denied names
   - Replaces `__import__` with `_create_safe_import()` wrapper

2. **`_sanitize_sys_modules()`**: Cleans `sys.modules` of non-allowed modules
   - Keeps essential modules (builtins, sys, traceback, importlib)
   - Keeps stdlib modules if `"*"` in stdlib_allow or module in allowlist
   - Keeps external modules if `"*"` in external_allow or module in allowlist
   - Removes everything else to prevent access to pre-imported modules

3. **`_create_safe_import()`**: Creates the import wrapper function
   - Validates each import against security config
   - Raises `SecurityViolationError` on blocked imports
   - Delegates to original `__import__` for allowed imports

=== Usage ===

These methods are called internally by `_all_items()` and `_per_item()` at the start of subprocess execution.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/task-runner-python/src/task_executor.py
* '''Lines:''' L424-477 (_filter_builtins, _sanitize_sys_modules), L479-495 (_create_safe_import)

=== Signature ===
<syntaxhighlight lang="python">
@staticmethod
def _filter_builtins(security_config: SecurityConfig) -> dict:
    """
    Get __builtins__ with denied ones removed.

    Args:
        security_config: Security configuration with builtins_deny set.

    Returns:
        Dictionary of filtered builtins with safe __import__.
    """

@staticmethod
def _sanitize_sys_modules(security_config: SecurityConfig) -> None:
    """
    Remove non-allowed modules from sys.modules.

    Args:
        security_config: Security configuration with allowlists.

    Side Effects:
        Modifies sys.modules in-place, removing disallowed entries.
    """

@staticmethod
def _create_safe_import(security_config: SecurityConfig):
    """
    Create an import wrapper that validates against security config.

    Args:
        security_config: Security configuration with allowlists.

    Returns:
        Wrapper function to replace __builtins__["__import__"].
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Internal methods - not directly imported
from src.task_executor import TaskExecutor
</syntaxhighlight>

== I/O Contract ==

=== _filter_builtins ===
{| class="wikitable"
|-
! Input !! Type !! Description
|-
| security_config || SecurityConfig || Contains builtins_deny set
|}

{| class="wikitable"
|-
! Output !! Type !! Description
|-
| (returns) || dict || Filtered builtins with safe __import__
|}

=== _sanitize_sys_modules ===
{| class="wikitable"
|-
! Input !! Type !! Description
|-
| security_config || SecurityConfig || Contains stdlib_allow and external_allow sets
|}

{| class="wikitable"
|-
! Side Effect !! Description
|-
| sys.modules modified || Non-allowed modules deleted
|}

== Usage Examples ==

=== Builtin Filtering ===
<syntaxhighlight lang="python">
@staticmethod
def _filter_builtins(security_config: SecurityConfig):
    if len(security_config.builtins_deny) == 0:
        # No filtering needed - copy original
        filtered = dict(__builtins__)
    else:
        # Filter out denied builtins
        filtered = {
            k: v
            for k, v in __builtins__.items()
            if k not in security_config.builtins_deny
        }

    # Always replace __import__ with safe wrapper
    filtered["__import__"] = TaskExecutor._create_safe_import(security_config)

    return filtered

# Example: with builtins_deny = {"eval", "exec", "compile", "open"}
# filtered will NOT contain eval, exec, compile, or open
# filtered["__import__"] will be the safe wrapper
</syntaxhighlight>

=== Module Sanitization ===
<syntaxhighlight lang="python">
@staticmethod
def _sanitize_sys_modules(security_config: SecurityConfig):
    # Essential modules always kept
    safe_modules = {
        "builtins",
        "__main__",
        "sys",
        "traceback",
        "linecache",
        "importlib",
        "importlib.machinery",
    }

    # Add allowed stdlib modules
    if "*" in security_config.stdlib_allow:
        safe_modules.update(sys.stdlib_module_names)
    else:
        safe_modules.update(security_config.stdlib_allow)

    # Add allowed external modules
    if "*" in security_config.external_allow:
        safe_modules.update(
            name for name in sys.modules.keys()
            if name not in sys.stdlib_module_names
        )
    else:
        safe_modules.update(security_config.external_allow)

    # Build safe prefixes for submodules
    safe_prefixes = [safe + "." for safe in safe_modules]

    # Remove everything not safe
    modules_to_remove = [
        name
        for name in sys.modules.keys()
        if name not in safe_modules
        and not any(name.startswith(prefix) for prefix in safe_prefixes)
    ]

    for module_name in modules_to_remove:
        del sys.modules[module_name]
</syntaxhighlight>

=== Safe Import Wrapper ===
<syntaxhighlight lang="python">
@staticmethod
def _create_safe_import(security_config: SecurityConfig):
    original_import = __builtins__["__import__"]

    def safe_import(name, *args, **kwargs):
        is_allowed, error_msg = validate_module_import(name, security_config)

        if not is_allowed:
            raise SecurityViolationError(
                message="Security violation detected",
                description=error_msg,
            )

        return original_import(name, *args, **kwargs)

    return safe_import

# Usage in sandbox:
# import os  → calls safe_import("os", ...)
#           → validate_module_import("os", config)
#           → (False, "Import 'os' is not allowed...")
#           → raises SecurityViolationError
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Sandbox_Environment]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Sandbox_Environment]]
