# Principle: Subprocess Creation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
* [[source::Doc|Python multiprocessing|https://docs.python.org/3/library/multiprocessing.html]]
|-
! Domains
| [[domain::Process_Isolation]], [[domain::Security]], [[domain::Code_Execution]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Principle for creating isolated subprocess environments to execute untrusted Python code with resource boundaries and communication channels.

=== Description ===

Subprocess Creation implements process-level isolation for code execution:

1. **ForkServer Context**: Uses `forkserver` multiprocessing context for clean process creation without copying parent state
2. **Pipe Communication**: Creates unidirectional pipe for result transmission from subprocess to parent
3. **Execution Mode Selection**: Chooses between "all_items" (batch) and "per_item" (streaming) execution functions
4. **Parameter Binding**: Passes code, input data, security config, and communication channel to subprocess

Process isolation provides:
- **Memory Isolation**: Subprocess has separate address space
- **Resource Limits**: Can be killed or timed out independently
- **Clean State**: No inherited file handles or connections from parent
- **Fault Containment**: Crashes in subprocess don't affect runner

=== Usage ===

Apply this principle when:
- Executing untrusted code that might crash or hang
- Implementing timeout-capable code execution
- Building multi-tenant systems requiring strong isolation
- Creating environments where resource consumption must be bounded

== Theoretical Basis ==

The subprocess creation follows a **Process Factory** pattern:

<syntaxhighlight lang="python">
# Pseudo-code for subprocess creation

def create_process(code, mode, items, security_config, query):
    # 1. Select execution function based on mode
    execution_fn = all_items_fn if mode == "all_items" else per_item_fn

    # 2. Create communication pipe (unidirectional)
    read_conn, write_conn = Pipe(duplex=False)
    # Parent reads, child writes

    # 3. Create process with target function and arguments
    process = ForkServerProcess(
        target=execution_fn,
        args=(code, items, write_conn, security_config, query)
    )

    return process, read_conn, write_conn
</syntaxhighlight>

Key design decisions:
- **ForkServer vs Fork**: ForkServer avoids copying parent's memory state
- **Unidirectional Pipe**: Simpler, prevents subprocess reading from parent
- **Function Selection**: Defers execution mode decision to subprocess

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_TaskExecutor_create_process]]
