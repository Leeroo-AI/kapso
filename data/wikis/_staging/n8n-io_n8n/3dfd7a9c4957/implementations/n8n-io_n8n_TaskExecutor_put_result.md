# Implementation: TaskExecutor._put_result / _put_error

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::IPC]], [[domain::Serialization]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Concrete static methods for serializing and transmitting execution results or errors from subprocess to parent via pipe.

=== Description ===

These methods handle the output pathway from the sandbox:

1. **`_put_result()`**: Sends successful execution results
   - Creates `PipeResultMessage` with result and print_args
   - Serializes to JSON with `default=str` for non-serializable values
   - Writes 4-byte length prefix followed by JSON data
   - Closes file descriptor after write

2. **`_put_error()`**: Sends execution errors
   - Creates `PipeErrorMessage` with `TaskErrorInfo` structure
   - Captures message, description, stack trace, and stderr
   - Handles `SystemExit` specially (reports exit code)
   - Same serialization and writing pattern

Both methods truncate print_args to `MAX_PRINT_ARGS_ALLOWED` (100) to prevent buffer overflow.

=== Usage ===

These methods are called internally by `_all_items()` and `_per_item()` to return results. They should not be called directly.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/task-runner-python/src/task_executor.py
* '''Lines:''' L298-351

=== Signature ===
<syntaxhighlight lang="python">
@staticmethod
def _put_result(write_fd: int, result: Items, print_args: PrintArgs):
    """
    Serialize and write successful result to pipe.

    Args:
        write_fd: File descriptor for pipe write end.
        result: Execution result (list of items).
        print_args: Captured print() call arguments.
    """

@staticmethod
def _put_error(
    write_fd: int,
    e: BaseException,
    stderr: str = "",
    print_args: PrintArgs | None = None,
):
    """
    Serialize and write error information to pipe.

    Args:
        write_fd: File descriptor for pipe write end.
        e: Exception that occurred during execution.
        stderr: Captured stderr output.
        print_args: Captured print() call arguments.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Internal methods - not directly imported
# Results are read via PipeReader in parent process
from src.task_executor import TaskExecutor
</syntaxhighlight>

== I/O Contract ==

=== _put_result Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| write_fd || int || Yes || File descriptor of pipe write end
|-
| result || Items || Yes || List of output items from execution
|-
| print_args || PrintArgs || Yes || List of formatted print() arguments
|}

=== _put_error Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| write_fd || int || Yes || File descriptor of pipe write end
|-
| e || BaseException || Yes || Exception that occurred
|-
| stderr || str || No || Captured stderr output (default: "")
|-
| print_args || PrintArgs || No || Captured print arguments (default: [])
|}

=== Wire Format ===
{| class="wikitable"
|-
! Bytes !! Content !! Description
|-
| 0-3 || Length (big-endian uint32) || Size of JSON payload in bytes
|-
| 4-N || JSON payload || UTF-8 encoded PipeResultMessage or PipeErrorMessage
|}

== Usage Examples ==

=== Result Message Structure ===
<syntaxhighlight lang="python">
# PipeResultMessage format
{
    "result": [
        {"json": {"key": "value"}, "pairedItem": {"item": 0}},
        {"json": {"key": "value2"}, "pairedItem": {"item": 1}}
    ],
    "print_args": [
        ["'Hello'", "123"],  # print("Hello", 123)
        ["{'key': 'value'}"]  # print({"key": "value"})
    ]
}
</syntaxhighlight>

=== Error Message Structure ===
<syntaxhighlight lang="python">
# PipeErrorMessage format
{
    "error": {
        "message": "division by zero",
        "description": "",
        "stack": "Traceback (most recent call last):\n  File ...",
        "stderr": ""
    },
    "print_args": []
}
</syntaxhighlight>

=== Wire Format Example ===
<syntaxhighlight lang="python">
# Example serialization
import json

result = [{"json": {"value": 42}}]
print_args = [["'Debug'"]]

message = {"result": result, "print_args": print_args}
data = json.dumps(message, ensure_ascii=False).encode("utf-8")

# data = b'{"result": [{"json": {"value": 42}}], "print_args": [["\'Debug\'"]]}'
# len(data) = 72 bytes
# length_bytes = b'\x00\x00\x00H'  # 72 in big-endian

# Wire: b'\x00\x00\x00H{"result": [{"json": ...'
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Result_Serialization]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Sandbox_Environment]]
