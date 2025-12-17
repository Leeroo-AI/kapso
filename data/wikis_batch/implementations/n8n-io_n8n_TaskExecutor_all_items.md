{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Automation]], [[domain::Task_Execution]], [[domain::Sandboxing]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Concrete tools for executing user Python code in sandboxed subprocess environments with two execution modes, provided by the n8n Python task runner.

=== Description ===

The `_all_items()` and `_per_item()` static methods execute user code in isolated subprocesses with security restrictions. `_all_items` processes all input items at once, while `_per_item` iterates through items individually. Both methods sanitize the environment, filter builtins, compile code, and send results back through pipes.

=== Usage ===

These implementations run inside forked subprocesses created by `create_process()`. They represent the actual execution environment where user code runs, with comprehensive security sandboxing including environment clearing, module filtering, and builtin restriction.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/task-runner-python/src/task_executor.py
* '''Lines:''' L185-278

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

@staticmethod
def _per_item(
    raw_code: str,
    items: Items,
    write_conn,
    security_config: SecurityConfig,
    _query: Query = None,  # unused
):
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from src.task_executor import TaskExecutor
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| raw_code || str || Yes || User Python code to execute
|-
| items || Items || Yes || Input data items
|-
| write_conn || PipeConnection || Yes || Pipe for sending results
|-
| security_config || SecurityConfig || Yes || Security policy
|-
| query || Query || Conditional || Used in all_items mode only
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| PipeResultMessage || dict || Success result with data and print args
|-
| PipeErrorMessage || dict || Error details with stack trace
|}

== Usage Examples ==

=== All-Items Mode Execution ===
<syntaxhighlight lang="python">
# This runs inside a subprocess, not called directly
# Example of what executes in all_items mode:

raw_code = """
result = []
for item in _items:
    value = item["json"]["value"]
    result.append({"json": {"doubled": value * 2}})
return result
"""

items = [
    {"json": {"value": 10}},
    {"json": {"value": 20}},
    {"json": {"value": 30}}
]

# Inside subprocess, this code has access to:
# - _items: All input items
# - _query: Query parameters
# - print(): Custom print for console logging
# - Filtered __builtins__

# Returns all results at once:
# [
#   {"json": {"doubled": 20}},
#   {"json": {"doubled": 40}},
#   {"json": {"doubled": 60}}
# ]
</syntaxhighlight>

=== Per-Item Mode Execution ===
<syntaxhighlight lang="python">
# Runs once per input item

raw_code = """
value = _item["json"]["value"]
if value > 15:
    return {"result": value, "category": "high"}
else:
    return {"result": value, "category": "low"}
"""

items = [
    {"json": {"value": 10}},
    {"json": {"value": 20}},
    {"json": {"value": 30}}
]

# Inside subprocess, each iteration has access to:
# - _item: Single item being processed
# - print(): Custom print for console logging
# - Filtered __builtins__

# Returns array with paired items:
# [
#   {"json": {"result": 10, "category": "low"}, "pairedItem": {"item": 0}},
#   {"json": {"result": 20, "category": "high"}, "pairedItem": {"item": 1}},
#   {"json": {"result": 30, "category": "high"}, "pairedItem": {"item": 2}}
# ]
</syntaxhighlight>

=== Using Print for Debugging ===
<syntaxhighlight lang="python">
raw_code = """
print("Starting processing...")
result = []
for item in _items:
    value = item["json"]["value"]
    print(f"Processing value: {value}")
    result.append({"json": {"output": value}})
print("Done processing")
return result
"""

# Print statements are captured and sent to browser console
# via RPC calls through the WebSocket connection
</syntaxhighlight>

== Implementation Details ==

=== Environment Sanitization ===
<syntaxhighlight lang="python">
if security_config.runner_env_deny:
    os.environ.clear()

TaskExecutor._sanitize_sys_modules(security_config)
</syntaxhighlight>

Security measures:
* '''Environment clearing:''' Removes all environment variables
* '''Module sanitization:''' Removes unauthorized modules from sys.modules
* '''Fresh imports:''' Subsequent imports use allowlist validation

=== Stderr Capture ===
<syntaxhighlight lang="python">
print_args: PrintArgs = []
sys.stderr = stderr_capture = io.StringIO()
</syntaxhighlight>

Captures error output for inclusion in error messages.

=== Code Wrapping and Compilation ===
<syntaxhighlight lang="python">
wrapped_code = TaskExecutor._wrap_code(raw_code)
compiled_code = compile(wrapped_code, EXECUTOR_ALL_ITEMS_FILENAME, "exec")

# _wrap_code transforms:
# user_code
#
# Into:
# def _user_function():
#     user_code
#
# _output = _user_function()
</syntaxhighlight>

Benefits of wrapping:
* Return statements work naturally
* Local scope for user variables
* Captures return value in _output

=== Global Namespace Construction (All-Items) ===
<syntaxhighlight lang="python">
globals = {
    "__builtins__": TaskExecutor._filter_builtins(security_config),
    "_items": items,
    "_query": query,
    "print": TaskExecutor._create_custom_print(print_args),
}

exec(compiled_code, globals)

result = globals[EXECUTOR_USER_OUTPUT_KEY]  # "_output"
</syntaxhighlight>

User code has access to:
* `_items`: List of all input items
* `_query`: Query parameters (if provided)
* `print()`: Custom print capturing output
* Filtered builtins (no eval, exec, etc.)

=== Global Namespace Construction (Per-Item) ===
<syntaxhighlight lang="python">
filtered_builtins = TaskExecutor._filter_builtins(security_config)
custom_print = TaskExecutor._create_custom_print(print_args)

result: Items = []
for index, item in enumerate(items):
    globals = {
        "__builtins__": filtered_builtins,
        "_item": item,
        "print": custom_print,
    }

    exec(compiled_code, globals)

    user_output = globals[EXECUTOR_USER_OUTPUT_KEY]

    if user_output is None:
        continue  # Skip items returning None

    # Extract JSON and add pairing info
    json_data = TaskExecutor._extract_json_data_per_item(user_output)

    output_item = {"json": json_data, "pairedItem": {"item": index}}

    if isinstance(user_output, dict) and "binary" in user_output:
        output_item["binary"] = user_output["binary"]

    result.append(output_item)
</syntaxhighlight>

Per-item processing:
* Executes code once per input item
* Skips items that return None
* Adds pairedItem tracking for workflow lineage
* Preserves binary data if present

=== Result Transmission ===
<syntaxhighlight lang="python">
TaskExecutor._put_result(write_conn.fileno(), result, print_args)

# _put_result sends length-prefixed JSON:
# [4 bytes: length] [JSON data]
message: PipeResultMessage = {
    "result": result,
    "print_args": TaskExecutor._truncate_print_args(print_args),
}
</syntaxhighlight>

=== Error Handling ===
<syntaxhighlight lang="python">
except BaseException as e:
    TaskExecutor._put_error(
        write_conn.fileno(), e, stderr_capture.getvalue(), print_args
    )

# _put_error sends:
message: PipeErrorMessage = {
    "error": {
        "message": str(e),
        "description": getattr(e, "description", ""),
        "stack": traceback.format_exc(),
        "stderr": stderr,
    },
    "print_args": print_args,
}
</syntaxhighlight>

Comprehensive error information:
* Exception message
* Optional description attribute
* Full stack trace
* Captured stderr output
* Print statements before error

=== Custom Print Implementation ===
<syntaxhighlight lang="python">
def _create_custom_print(print_args: PrintArgs):
    def custom_print(*args):
        serializable_args = []

        for arg in args:
            try:
                json.dumps(arg, default=str, ensure_ascii=False)
                serializable_args.append(arg)
            except Exception:
                # Handle circular references
                serializable_args.append({
                    EXECUTOR_CIRCULAR_REFERENCE_KEY: repr(arg),
                    "__type__": type(arg).__name__,
                })

        formatted = TaskExecutor._format_print_args(*serializable_args)
        print_args.append(formatted)
        print("[user code]", *args)  # Also print to subprocess stdout

    return custom_print
</syntaxhighlight>

Custom print features:
* Captures arguments for browser console
* Handles circular references gracefully
* Formats for JavaScript console display
* Still prints to subprocess stdout for debugging

=== Builtin Filtering ===
<syntaxhighlight lang="python">
def _filter_builtins(security_config: SecurityConfig):
    if len(security_config.builtins_deny) == 0:
        filtered = dict(__builtins__)
    else:
        filtered = {
            k: v
            for k, v in __builtins__.items()
            if k not in security_config.builtins_deny
        }

    # Replace __import__ with safe version
    filtered["__import__"] = TaskExecutor._create_safe_import(security_config)

    return filtered
</syntaxhighlight>

Common denied builtins:
* `eval`: Dynamic code execution
* `exec`: Dynamic code execution
* `compile`: Code compilation
* `open`: File system access
* `__import__`: Replaced with allowlist-checking version

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Code_Execution]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Task_Runner]]

=== Related Implementations ===
* [[Implementation:n8n-io_n8n_TaskExecutor_create_process]]
* [[Implementation:n8n-io_n8n_TaskExecutor_execute_process]]
* [[Implementation:n8n-io_n8n_TaskAnalyzer_validate]]

=== Execution Modes Comparison ===
{| class="wikitable"
|-
! Aspect !! All-Items !! Per-Item
|-
| Input variable || _items (list) || _item (single dict)
|-
| Query support || Yes (_query) || No
|-
| Execution count || Once || Once per item
|-
| Return value || List of items || Single item or None
|-
| Pairing info || Not added || Added automatically
|-
| Use case || Bulk operations || Item-by-item transformations
|}

=== Security Layers ===
# Environment variable clearing
# sys.modules sanitization
# Builtin filtering (remove eval, exec, etc.)
# Custom __import__ with allowlist checking
# Isolated subprocess execution
# Limited sys module access
