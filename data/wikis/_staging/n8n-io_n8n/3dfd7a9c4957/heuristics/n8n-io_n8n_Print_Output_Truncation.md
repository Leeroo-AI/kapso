# Heuristic: Print Output Truncation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n-io/n8n|https://github.com/n8n-io/n8n]]
* [[source::Code|task_executor.py|packages/@n8n/task-runner-python/src/task_executor.py]]
|-
! Domains
| [[domain::Performance]], [[domain::Resource_Limits]], [[domain::User_Experience]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==
Limit captured print() calls to 100 statements to prevent pipe buffer overflow and excessive memory usage from verbose user code.

=== Description ===
User Python code can include unlimited `print()` calls for debugging. These are captured and transmitted back to the n8n UI for display. Without limits, a loop with print statements could generate millions of entries, causing memory exhaustion or pipe buffer overflow. This heuristic truncates output at 100 entries with a clear message indicating truncation.

=== Usage ===
This heuristic is automatically applied by the `TaskExecutor`. No configuration is needed. If users see "[Output truncated - N more print statements]", they should reduce print verbosity or use logging to files.

== The Insight (Rule of Thumb) ==

* **Action:** Truncate `print_args` list to `MAX_PRINT_ARGS_ALLOWED` entries
* **Value:** `MAX_PRINT_ARGS_ALLOWED = 100` print statements
* **Trade-off:** May lose debugging output from verbose code, but prevents OOM and pipe overflow
* **User feedback:** Truncation message shows exactly how many statements were dropped

== Reasoning ==

1. **Unbounded print() is dangerous:**
   ```python
   for i in range(1000000):
       print(i)  # Would create 1M entries without truncation
   ```

2. **Pipe buffer limits:** The IPC pipe has finite buffer space; excessive data blocks the subprocess

3. **JSON serialization cost:** Each print argument must be JSON-serialized; 1M entries = significant memory

4. **100 is a reasonable limit:** Enough for debugging, not enough to cause problems

5. **Clear feedback:** Users know output was truncated and by how much

== Code Evidence ==

From `task_executor.py:48`:

<syntaxhighlight lang="python">
MAX_PRINT_ARGS_ALLOWED = 100
</syntaxhighlight>

From `task_executor.py:406-420`:

<syntaxhighlight lang="python">
@staticmethod
def _truncate_print_args(print_args: PrintArgs) -> PrintArgs:
    """Truncate print_args to prevent pipe buffer overflow."""

    if not print_args or len(print_args) <= MAX_PRINT_ARGS_ALLOWED:
        return print_args

    truncated = print_args[:MAX_PRINT_ARGS_ALLOWED]
    truncated.append(
        [
            f"[Output truncated - {len(print_args) - MAX_PRINT_ARGS_ALLOWED} more print statements]"
        ]
    )

    return truncated
</syntaxhighlight>

Applied in result serialization from `task_executor.py:302`:

<syntaxhighlight lang="python">
message: PipeResultMessage = {
    "result": result,
    "print_args": TaskExecutor._truncate_print_args(print_args),
}
</syntaxhighlight>

== Related Pages ==

* [[uses_heuristic::Implementation:n8n-io_n8n_TaskExecutor_put_result]]
* [[uses_heuristic::Implementation:n8n-io_n8n_TaskExecutor_execute]]
* [[uses_heuristic::Workflow:n8n-io_n8n_Python_Task_Execution]]
