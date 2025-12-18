# Implementation: TaskExecutor.create_process

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Process_Isolation]], [[domain::Code_Execution]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Concrete static method for creating an isolated subprocess with communication pipe for executing Python code tasks.

=== Description ===

`TaskExecutor.create_process()` creates the subprocess infrastructure for code execution:

1. **Mode-Based Function Selection**: Selects `_all_items` or `_per_item` based on `node_mode` parameter
2. **Pipe Creation**: Creates unidirectional pipe using forkserver context (`MULTIPROCESSING_CONTEXT.Pipe(duplex=False)`)
3. **Process Creation**: Creates `ForkServerProcess` targeting the selected execution function
4. **Argument Binding**: Passes code, items, write connection, security config, and optional query to process

The method returns a tuple allowing the caller to manage process lifecycle and read results.

=== Usage ===

Call this method to prepare a subprocess for execution. The returned process must be started with `process.start()` and results read from `read_conn`.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/task-runner-python/src/task_executor.py
* '''Lines:''' L56-86

=== Signature ===
<syntaxhighlight lang="python">
@staticmethod
def create_process(
    code: str,
    node_mode: NodeMode,
    items: Items,
    security_config: SecurityConfig,
    query: Query = None,
) -> tuple[ForkServerProcess, PipeConnection, PipeConnection]:
    """
    Create a subprocess for executing a Python code task and a pipe for communication.

    Args:
        code: Python source code to execute.
        node_mode: "all_items" for batch execution, "per_item" for streaming.
        items: List of input items to process.
        security_config: Security configuration with allowlists/denylists.
        query: Optional query data for database access.

    Returns:
        Tuple of (process, read_connection, write_connection).
        - process: ForkServerProcess ready to be started
        - read_connection: Parent reads results from here
        - write_connection: Subprocess writes results here (close after start)
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from src.task_executor import TaskExecutor
from src.config.security_config import SecurityConfig
from src.message_types.broker import NodeMode, Items, Query
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| code || str || Yes || Python source code to execute
|-
| node_mode || NodeMode || Yes || "all_items" or "per_item" execution mode
|-
| items || Items || Yes || List of input items (dicts with json/binary keys)
|-
| security_config || SecurityConfig || Yes || Security policy configuration
|-
| query || Query || No || Optional query data for database operations
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| process || ForkServerProcess || Subprocess ready to execute, not yet started
|-
| read_conn || PipeConnection || Connection for parent to read results
|-
| write_conn || PipeConnection || Connection for subprocess to write results (close after process.start())
|}

== Usage Examples ==

=== Basic Process Creation ===
<syntaxhighlight lang="python">
from src.task_executor import TaskExecutor
from src.config.security_config import SecurityConfig

# Configure security
security_config = SecurityConfig(
    stdlib_allow={"json"},
    external_allow=set(),
    builtins_deny=set(),
    runner_env_deny=True,
)

# Sample code and items
code = "return [item['json'] for item in _items]"
items = [{"json": {"value": 1}}, {"json": {"value": 2}}]

# Create process
process, read_conn, write_conn = TaskExecutor.create_process(
    code=code,
    node_mode="all_items",
    items=items,
    security_config=security_config,
)

# Start process and close write connection
process.start()
write_conn.close()

# Read result
process.join(timeout=10)
# Result available via read_conn...
</syntaxhighlight>

=== Per-Item Mode ===
<syntaxhighlight lang="python">
# Per-item mode processes each item individually
code = "return {'value': _item['json']['value'] * 2}"
items = [{"json": {"value": 1}}, {"json": {"value": 2}}]

process, read_conn, write_conn = TaskExecutor.create_process(
    code=code,
    node_mode="per_item",  # Process each item separately
    items=items,
    security_config=security_config,
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Subprocess_Creation]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Task_Runner_Env]]
