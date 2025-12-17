{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Automation]], [[domain::Task_Execution]], [[domain::Process_Isolation]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Concrete tool for creating isolated subprocesses for Python code execution with inter-process communication, provided by the n8n Python task runner.

=== Description ===

The `create_process()` static method creates a subprocess using Python's forkserver context for secure task execution. It sets up a unidirectional pipe for communication, selects the appropriate execution function based on node mode, and returns the process handle and communication endpoints.

=== Usage ===

This implementation is invoked during task execution to create an isolated environment for running user Python code. It establishes the foundation for secure execution by using the forkserver multiprocessing context and setting up communication channels.

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
| items || Items || Yes || Input data items to process
|-
| security_config || SecurityConfig || Yes || Security policy configuration
|-
| query || Query || No || Optional query parameters for all_items mode
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| process || ForkServerProcess || Subprocess handle for execution
|-
| read_conn || PipeConnection || Read end of pipe (runner reads)
|-
| write_conn || PipeConnection || Write end of pipe (subprocess writes)
|}

== Usage Examples ==

=== Creating Process for All-Items Mode ===
<syntaxhighlight lang="python">
from src.task_executor import TaskExecutor
from src.config.security_config import SecurityConfig

code = """
result = []
for item in _items:
    result.append({"json": item["json"]["value"] * 2})
return result
"""

items = [
    {"json": {"value": 10}},
    {"json": {"value": 20}}
]

security_config = SecurityConfig(
    stdlib_allow=["json", "math"],
    external_allow=["numpy"],
    builtins_deny=["eval", "exec"],
    runner_env_deny=True
)

# Create subprocess for all-items mode
process, read_conn, write_conn = TaskExecutor.create_process(
    code=code,
    node_mode="all_items",
    items=items,
    security_config=security_config,
    query={"param": "value"}
)

# Process is ready but not started yet
assert not process.is_alive()
</syntaxhighlight>

=== Creating Process for Per-Item Mode ===
<syntaxhighlight lang="python">
from src.task_executor import TaskExecutor

code = """
# Process single item
value = _item["json"]["value"]
return {"result": value * 3}
"""

items = [
    {"json": {"value": 5}},
    {"json": {"value": 10}},
    {"json": {"value": 15}}
]

# Create subprocess for per-item mode
process, read_conn, write_conn = TaskExecutor.create_process(
    code=code,
    node_mode="per_item",
    items=items,
    security_config=security_config
    # query parameter ignored in per-item mode
)

# Different execution function selected based on node_mode
</syntaxhighlight>

== Implementation Details ==

=== Forkserver Context ===
<syntaxhighlight lang="python">
MULTIPROCESSING_CONTEXT = multiprocessing.get_context("forkserver")
</syntaxhighlight>

The forkserver context provides:
* '''Security:''' Fresh process without parent's memory
* '''Isolation:''' Minimal inherited state
* '''Predictability:''' Consistent subprocess behavior

Comparison with other contexts:
* '''fork:''' Copies parent's entire memory (security risk)
* '''spawn:''' Slower startup, Windows compatible
* '''forkserver:''' Best balance for Linux/Unix systems

=== Execution Function Selection ===
<syntaxhighlight lang="python">
fn = (
    TaskExecutor._all_items
    if node_mode == "all_items"
    else TaskExecutor._per_item
)
</syntaxhighlight>

Two execution modes:
* '''all_items:''' Receives all items at once, returns bulk results
* '''per_item:''' Iterates through items, processes individually

=== Pipe Creation ===
<syntaxhighlight lang="python">
# thread in runner process reads, subprocess writes
read_conn, write_conn = MULTIPROCESSING_CONTEXT.Pipe(duplex=False)
</syntaxhighlight>

Unidirectional pipe characteristics:
* '''Simplex communication:''' Subprocess → Runner only
* '''Thread-safe:''' Runner reads from separate thread
* '''Blocking reads:''' Runner waits for subprocess output
* '''Buffered:''' Can hold data until read

=== Process Creation ===
<syntaxhighlight lang="python">
process = MULTIPROCESSING_CONTEXT.Process(
    target=fn,
    args=(
        code,
        items,
        write_conn,
        security_config,
        query,
    ),
)
</syntaxhighlight>

Process is created but not started:
* No resources allocated yet
* No execution begun
* Can be inspected or configured before start

=== Arguments Passed to Subprocess ===
{| class="wikitable"
|-
! Argument !! Purpose !! Used By
|-
| code || Python source code to execute || Both modes
|-
| items || Input data items || Both modes
|-
| write_conn || Pipe for sending results back || Both modes
|-
| security_config || Security policy enforcement || Both modes
|-
| query || Query parameters || all_items only
|}

=== Return Value Usage ===
<syntaxhighlight lang="python">
# Caller starts process and manages lifecycle
process.start()  # Begin execution
process.join(timeout=30)  # Wait for completion

# Read results from pipe
result_data = read_conn.recv()

# Cleanup
read_conn.close()
write_conn.close()
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Subprocess_Isolation]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Task_Runner]]

=== Related Implementations ===
* [[Implementation:n8n-io_n8n_TaskExecutor_execute_process]]
* [[Implementation:n8n-io_n8n_TaskExecutor_all_items]]
* [[Implementation:n8n-io_n8n_TaskRunner_execute_task]]

=== Multiprocessing Context ===
* '''Context type:''' forkserver
* '''Platform:''' Unix/Linux systems
* '''Security benefit:''' Process starts with minimal state
* '''Trade-off:''' Slightly slower than fork, faster than spawn

=== Communication Architecture ===
* '''Pipe type:''' Unidirectional (simplex)
* '''Direction:''' Subprocess → Runner process
* '''Data format:''' JSON serialized results
* '''Protocol:''' Length-prefixed messages
