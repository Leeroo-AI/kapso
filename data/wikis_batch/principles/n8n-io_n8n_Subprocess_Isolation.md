{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|n8n Python Task Runner|https://docs.n8n.io]]
|-
! Domains
| [[domain::Task_Execution]], [[domain::Security]], [[domain::Process_Management]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Process-level isolation creates independent subprocesses for executing untrusted code with memory separation, resource limits, and sanitized environments.

=== Description ===

Subprocess isolation leverages operating system process boundaries to create security and fault isolation when executing untrusted Python code. Each task runs in a separate subprocess with:
* **Memory Isolation**: Separate address space prevents access to parent process memory
* **Resource Limits**: OS-enforced CPU and memory quotas prevent resource exhaustion
* **Clean Environment**: Sanitized environment variables and file descriptors
* **Crash Isolation**: Subprocess crashes (segfaults, OOM) don't affect parent
* **Signal Control**: Parent can terminate misbehaving subprocesses via SIGTERM/SIGKILL

The implementation uses Python's multiprocessing module with the "forkserver" context, which provides a clean process initialization model without inheriting unnecessary state from the parent. Communication occurs through pipes, providing unidirectional data flow from child to parent.

This approach provides defense-in-depth: even if code bypasses static analysis or runtime sandboxing, process boundaries limit blast radius.

=== Usage ===

Apply this principle when executing untrusted or user-provided code where:
* Code may contain bugs leading to crashes or resource exhaustion
* Memory isolation is required to protect sensitive data in parent process
* Individual task failures shouldn't compromise system availability
* OS-level security boundaries provide stronger guarantees than language-level sandboxing
* Resource limits (CPU, memory, time) must be enforced reliably
* System needs to survive malicious code attempts (fork bombs, memory leaks)
* Clean termination of misbehaving code is critical

== Theoretical Basis ==

Subprocess isolation uses **operating system process boundaries** for security and fault tolerance:

**Process Isolation Properties:**

1. **Memory Isolation**:
   ```
   Parent Process: Memory space [0x1000 - 0x2000]
   Child Process:  Memory space [0x3000 - 0x4000]
   Child cannot read/write parent memory (enforced by MMU)
   ```

2. **Resource Limits** (via OS mechanisms):
   ```python
   setrlimit(RLIMIT_CPU, soft=60)      # CPU time limit
   setrlimit(RLIMIT_AS, soft=512MB)    # Memory limit
   setrlimit(RLIMIT_NPROC, soft=1)     # Process count limit
   ```

3. **Namespace Isolation**:
   ```
   Parent globals: {task_runner_state, active_tasks, ...}
   Child globals:  {code_to_execute, items, ...}
   No shared state except explicit IPC
   ```

**Forkserver Context:**

Traditional fork() copies entire parent process (copy-on-write), including:
* Open file descriptors
* Thread state
* Loaded libraries
* Signal handlers

Forkserver creates a minimal clean process at startup, then forks from that:
```
Startup: Main Process → Forkserver (clean minimal process)
Task Execution: Forkserver → fork() → Task Subprocess
```

Benefits:
* Predictable process state
* No inherited locks or threads
* Minimal memory footprint
* Faster fork times

**Inter-Process Communication:**

```python
parent_conn, child_conn = multiprocessing.Pipe()
Process isolation via:
  - Unidirectional pipe (child → parent)
  - Serialized data (pickle protocol)
  - No shared memory
```

**Termination Semantics:**

```
Normal: subprocess.wait(timeout) → exit code 0
Timeout: send SIGTERM → wait(grace_period) → send SIGKILL
Crash: wait() returns signal number (e.g., SIGSEGV)
```

**Security Properties:**

* **Containment**: Code cannot escape subprocess boundary
* **Least Privilege**: Subprocess has minimal permissions
* **Fail-Safe**: Crashes contained, parent continues
* **Time-Bounded**: Enforced execution timeouts
* **Resource-Bounded**: OS-enforced limits

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_TaskExecutor_create_process]]

=== Related Principles ===
* [[Principle:n8n-io_n8n_Static_Security_Analysis]] - Complements runtime isolation
* [[Principle:n8n-io_n8n_Code_Execution]] - Runs within isolated subprocess
* [[Principle:n8n-io_n8n_Result_Collection]] - Retrieves results via IPC
