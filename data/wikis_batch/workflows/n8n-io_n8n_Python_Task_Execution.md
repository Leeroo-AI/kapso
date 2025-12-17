{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
* [[source::Doc|n8n Docs|https://docs.n8n.io]]
|-
! Domains
| [[domain::Workflow_Automation]], [[domain::Code_Execution]], [[domain::Security]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

End-to-end process for securely executing user-provided Python code within n8n workflows using isolated subprocess execution.

=== Description ===

This workflow describes the Python Task Runner's core execution flow. The Task Runner is a WebSocket-connected service that receives Python code tasks from a broker, validates them for security compliance, executes them in isolated subprocesses, and returns results. The system implements defense-in-depth security with static AST analysis and runtime import validation, subprocess isolation using the forkserver multiprocessing context, and offer-based task distribution for load balancing.

=== Usage ===

Execute this workflow when:
* A user defines a Python Code node in an n8n workflow that needs execution
* The n8n execution engine dispatches a Python task to the task broker
* The broker needs to distribute code execution work to available runners

This is the primary "golden path" for all Python code execution within n8n workflows.

== Execution Steps ==

=== Step 1: WebSocket Connection & Registration ===
[[step::Principle:n8n-io_n8n_WebSocket_Connection]]

The Task Runner establishes a persistent WebSocket connection to the broker and registers itself. Upon connection, the broker sends an info request, and the runner responds with its capabilities (name, supported task types). After registration confirmation, the runner is authorized to send task offers.

'''Key behaviors:'''
* Automatic reconnection with 5-second backoff on connection failure
* Bearer token authentication via grant token
* Connection uses configurable max payload size

=== Step 2: Task Offer Distribution ===
[[step::Principle:n8n-io_n8n_Offer_Based_Distribution]]

The runner advertises available capacity by periodically sending task offers to the broker. Each offer has a unique ID and validity window. The broker selects runners based on available offers, enabling load balancing across multiple runners.

'''Offer mechanics:'''
* Offers sent at regular intervals (configurable)
* Each offer has validity timeout with jitter to prevent thundering herd
* Number of concurrent offers = max_concurrency - (open_offers + running_tasks)
* Expired offers are cleaned up before sending new ones

=== Step 3: Task Acceptance ===
[[step::Principle:n8n-io_n8n_Task_Acceptance]]

When the broker accepts an offer, it sends a task assignment with the task ID. The runner validates the offer hasn't expired and capacity is available, then either accepts (creating task state) or rejects (with reason: expired/at_capacity).

'''State transition:'''
* New task enters WAITING_FOR_SETTINGS state
* Runner sends accepted/rejected response to broker
* Task settings (code, items, mode) arrive in subsequent message

=== Step 4: Static Security Analysis ===
[[step::Principle:n8n-io_n8n_Static_Security_Analysis]]

Before execution, the code undergoes AST-based static analysis to detect security violations. The analyzer parses the code into an AST and visits nodes to check for blocked imports, dangerous attribute access, and relative imports.

'''Security checks performed:'''
* Import statements validated against stdlib/external allowlists
* Blocked attributes (e.g., __code__, __globals__) are rejected
* Name-mangled attributes (e.g., _ClassName__attr) are blocked
* Dynamic __import__() calls with non-literal arguments are rejected
* Validation results are cached (LRU, 500 entries) by code hash

=== Step 5: Subprocess Creation ===
[[step::Principle:n8n-io_n8n_Subprocess_Isolation]]

The executor creates an isolated subprocess using Python's forkserver multiprocessing context. A unidirectional pipe is established for result communication (subprocess writes, runner reads).

'''Isolation setup:'''
* Forkserver context chosen for better isolation than fork
* Code wrapped in function to capture return value
* Custom builtins with blocked items removed
* Safe import wrapper installed for runtime validation
* Environment variables optionally cleared for security

=== Step 6: Code Execution ===
[[step::Principle:n8n-io_n8n_Code_Execution]]

The subprocess executes the user code in either "all_items" or "per_item" mode. The code receives input items and optional query parameters. A custom print() function captures console output for relay to the browser.

'''Execution modes:'''
* all_items: Code receives all items at once, returns list of output items
* per_item: Code runs once per input item, results aggregated
* Both modes support continue_on_fail option

=== Step 7: Result Collection ===
[[step::Principle:n8n-io_n8n_Result_Collection]]

Results are collected via the IPC pipe using length-prefixed JSON messages. A background PipeReader thread reads results asynchronously while the main thread monitors subprocess status and handles timeouts.

'''Result handling:'''
* Length prefix (4 bytes big-endian) followed by JSON payload
* Results include both output data and captured print() arguments
* Timeout triggers graceful termination (SIGTERM), then force kill (SIGKILL)
* Print arguments truncated to prevent pipe buffer overflow

=== Step 8: Task Completion ===
[[step::Principle:n8n-io_n8n_Task_Completion]]

The runner sends task results (success or error) back to the broker via WebSocket. Print output is sent as RPC calls to display in the browser console. Task state is cleaned up and idle timer reset.

'''Completion behaviors:'''
* RunnerTaskDone message sent on success with result data
* RunnerTaskError message sent on failure with error details
* RPC calls relay captured print() output to browser
* Task removed from running_tasks map

== Execution Diagram ==

{{#mermaid:graph TD
    A[WebSocket Connection] --> B[Runner Registration]
    B --> C[Send Task Offers]
    C --> D{Offer Accepted?}
    D -->|No| C
    D -->|Yes| E[Accept Task]
    E --> F[Static Security Analysis]
    F -->|Violation| G[Return Error]
    F -->|Pass| H[Create Subprocess]
    H --> I[Execute Code]
    I --> J[Collect Results via Pipe]
    J --> K{Success?}
    K -->|Yes| L[Send TaskDone]
    K -->|No| G
    G --> M[Send TaskError]
    L --> N[Reset Idle Timer]
    M --> N
    N --> C
}}

== Related Pages ==

* [[step::Principle:n8n-io_n8n_WebSocket_Connection]]
* [[step::Principle:n8n-io_n8n_Offer_Based_Distribution]]
* [[step::Principle:n8n-io_n8n_Task_Acceptance]]
* [[step::Principle:n8n-io_n8n_Static_Security_Analysis]]
* [[step::Principle:n8n-io_n8n_Subprocess_Isolation]]
* [[step::Principle:n8n-io_n8n_Code_Execution]]
* [[step::Principle:n8n-io_n8n_Result_Collection]]
* [[step::Principle:n8n-io_n8n_Task_Completion]]
