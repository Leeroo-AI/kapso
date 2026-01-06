# Implementation: TaskExecutor._all_items / _per_item

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Security]], [[domain::Code_Execution]], [[domain::Sandboxing]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Concrete static methods for executing user Python code within a sandboxed subprocess environment with two execution modes.

=== Description ===

These methods are the subprocess entry points that execute user code:

1. **`_all_items()`**: Batch mode - code receives all items at once via `_items` variable
2. **`_per_item()`**: Streaming mode - code executes once per item via `_item` variable

Both methods:
- Clear environment variables if `runner_env_deny` is True
- Sanitize `sys.modules` to remove non-allowlisted modules
- Create filtered `__builtins__` with safe import wrapper
- Compile and execute wrapped user code
- Capture results via pipe or report errors

=== Usage ===

These methods are called as subprocess targets. They should not be called directly - use `create_process()` and `execute_process()` instead.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/task-runner-python/src/task_executor.py
* '''Lines:''' L186-278

=== Signature ===
<syntaxhighlight lang="python">
@staticmethod
def _all_items(
    raw_code: str,
    items: Items,
    write_conn,
    security_config: SecurityConfig,
    query: Query = None,
):
    """
    Execute a Python code task in all-items mode.

    User code receives:
        _items: List of all input items
        _query: Optional query data
        print(): Custom print that captures output

    User code should return the result directly.
    """

@staticmethod
def _per_item(
    raw_code: str,
    items: Items,
    write_conn,
    security_config: SecurityConfig,
    _query: Query = None,
):
    """
    Execute a Python code task in per-item mode.

    User code receives:
        _item: Current item being processed
        print(): Custom print that captures output

    User code executes once per item, results are aggregated.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# These are internal subprocess targets, not directly imported
# Use TaskExecutor.create_process() instead
from src.task_executor import TaskExecutor
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| raw_code || str || Yes || User's Python source code
|-
| items || Items || Yes || Input items to process
|-
| write_conn || PipeConnection || Yes || Pipe connection for result transmission
|-
| security_config || SecurityConfig || Yes || Security policy configuration
|-
| query || Query || No || Optional query data (all_items only)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| (via pipe) || PipeResultMessage || JSON with 'result' key containing processed items
|-
| (via pipe) || PipeErrorMessage || JSON with 'error' key if execution fails
|}

== Usage Examples ==

=== Code Wrapping ===
<syntaxhighlight lang="python">
# User code is wrapped to capture return value
# Input:
"""
return [x * 2 for x in _items]
"""

# Becomes:
"""
def _user_function():
    return [x * 2 for x in _items]

_output = _user_function()
"""
</syntaxhighlight>

=== All-Items Mode Globals ===
<syntaxhighlight lang="python">
# Globals provided to all_items execution
globals = {
    "__builtins__": filtered_builtins,  # Restricted builtins
    "_items": items,                     # All input items
    "_query": query,                     # Optional query data
    "print": custom_print,               # Captured print function
}
</syntaxhighlight>

=== Per-Item Mode Globals ===
<syntaxhighlight lang="python">
# Globals provided to per_item execution (per iteration)
globals = {
    "__builtins__": filtered_builtins,
    "_item": items[index],               # Current item
    "print": custom_print,
}

# Result aggregation
result = []
for index, item in enumerate(items):
    # Execute code...
    if user_output is not None:
        result.append({
            "json": extracted_json,
            "pairedItem": {"item": index}
        })
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Sandboxed_Execution]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Sandbox_Environment]]
