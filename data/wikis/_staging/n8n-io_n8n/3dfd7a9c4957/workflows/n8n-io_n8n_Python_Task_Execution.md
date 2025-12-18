# Python Task Execution

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
* [[source::Doc|n8n Docs|https://docs.n8n.io]]
|-
! Domains
| [[domain::Workflow_Automation]], [[domain::Task_Execution]], [[domain::Python]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:30 GMT]]
|}

== Overview ==

End-to-end process for securely executing Python code tasks in n8n's distributed task runner system, from broker communication through sandboxed execution to result delivery.

=== Description ===

This workflow documents the complete lifecycle of a Python task in n8n's task runner infrastructure. The task runner is a distributed system where:

1. **Goal:** Execute user-provided Python code in isolated subprocesses and return structured results to the n8n workflow engine.
2. **Scope:** Covers WebSocket connection to the broker, task offer/acceptance negotiation, security validation, subprocess creation, code execution, and result serialization.
3. **Strategy:** Uses a broker-runner architecture with forkserver-based process isolation, AST-based security validation, and pipe-based IPC for reliable result transmission.

The system supports two execution modes:
- **All Items Mode:** Process all workflow items in a single execution context
- **Per Item Mode:** Process each item independently with paired item tracking

=== Usage ===

Execute this workflow when:
- You need to understand how Python code nodes execute within n8n workflows
- You are debugging task execution issues in the Python task runner
- You want to extend or modify the task runner's execution behavior
- You need to trace a task from receipt to completion for troubleshooting

== Execution Steps ==

=== Step 1: Runner Initialization ===
[[step::Principle:n8n-io_n8n_Runner_Initialization]]

Initialize the Python task runner with configuration from environment variables. This step sets up the WebSocket connection parameters, security configuration (allowed modules, blocked builtins), concurrency limits, and optional components like health check server and Sentry error tracking.

'''Key considerations:'''
* Configuration is loaded from N8N_RUNNERS_* environment variables
* Security config defines stdlib_allow, external_allow, and builtins_deny lists
* Health check server enables Kubernetes liveness/readiness probes
* Auto-shutdown timeout can be configured for ephemeral runners

=== Step 2: Broker Connection ===
[[step::Principle:n8n-io_n8n_Broker_Connection]]

Establish and maintain a WebSocket connection to the n8n task broker. The runner authenticates using a grant token and registers its capabilities (task type: "python"). The connection includes automatic reconnection logic with exponential backoff.

'''Pseudocode:'''
  1. Connect to broker WebSocket endpoint with auth header
  2. Wait for BrokerInfoRequest message
  3. Send RunnerInfo with name and supported task types
  4. Receive BrokerRunnerRegistered confirmation
  5. Start task offer loop

=== Step 3: Task Offer Negotiation ===
[[step::Principle:n8n-io_n8n_Task_Offer_Negotiation]]

Implement the task offer/acceptance protocol to claim work from the broker. The runner sends periodic offers indicating available capacity, and the broker accepts offers when tasks are queued. This decoupled model allows multiple runners to load-balance task execution.

'''What happens:'''
* Runner sends RunnerTaskOffer messages at regular intervals (250ms)
* Each offer has a validity window (5s with jitter) to prevent stale accepts
* Broker responds with BrokerTaskOfferAccept containing task_id
* Runner validates offer hasn't expired before accepting
* Rejected offers return reason: "at capacity" or "offer expired"

=== Step 4: Security Validation ===
[[step::Principle:n8n-io_n8n_Security_Validation]]

Validate the task's Python code against security policies before execution. This step uses AST-based static analysis to detect disallowed imports, dangerous attribute access, and security bypass attempts. Results are cached by code hash for performance.

'''Key considerations:'''
* Validates import statements against stdlib_allow and external_allow lists
* Blocks dangerous names: __loader__, __builtins__, __globals__, etc.
* Blocks dangerous attributes: __subclasses__, __code__, f_globals, etc.
* Detects dynamic __import__() calls and relative imports
* Cache key includes code hash + allowlist tuple for invalidation

=== Step 5: Subprocess Creation ===
[[step::Principle:n8n-io_n8n_Subprocess_Creation]]

Create an isolated subprocess for code execution using Python's forkserver multiprocessing context. This provides process-level isolation between tasks and prevents memory leaks from accumulating across executions.

'''Pseudocode:'''
  1. Select execution function (all_items or per_item based on mode)
  2. Create unidirectional pipe for result communication
  3. Spawn forkserver process with code, items, security config
  4. Return process handle and pipe connections

=== Step 6: Sandboxed Code Execution ===
[[step::Principle:n8n-io_n8n_Sandboxed_Execution]]

Execute the Python code within a restricted sandbox environment. The sandbox filters builtins, sanitizes sys.modules, intercepts imports for validation, and provides custom print() for output capture.

'''What happens:'''
* Environment variables cleared if runner_env_deny is set
* sys.modules sanitized to only include allowed modules
* __builtins__ filtered to remove dangerous functions (eval, exec, open, etc.)
* __import__ replaced with validating wrapper
* User code wrapped in function for clean namespace isolation
* Custom print() captures arguments for browser console forwarding

=== Step 7: Result Serialization ===
[[step::Principle:n8n-io_n8n_Result_Serialization]]

Serialize execution results and transmit them back to the runner process via pipe. Results are JSON-encoded with length-prefixed framing for reliable transmission.

'''Key considerations:'''
* Results serialized with json.dumps(default=str) for safety
* Circular references converted to special marker objects
* Print arguments truncated to MAX_PRINT_ARGS_ALLOWED (100)
* Length prefix is 4 bytes (big-endian) allowing up to 4GB results
* Errors serialized with message, description, stack trace, and stderr

=== Step 8: Result Delivery ===
[[step::Principle:n8n-io_n8n_Result_Delivery]]

Send the execution result back to the broker and clean up task state. The runner sends either RunnerTaskDone with results or RunnerTaskError with error details.

'''Pseudocode:'''
  1. Read result from pipe with timeout
  2. Forward print() calls as RPC messages to browser console
  3. Send RunnerTaskDone or RunnerTaskError to broker
  4. Remove task from running_tasks
  5. Reset idle timer for auto-shutdown

== Execution Diagram ==

{{#mermaid:graph TD
    A[Runner Initialization] --> B[Broker Connection]
    B --> C[Task Offer Negotiation]
    C --> D[Security Validation]
    D --> E[Subprocess Creation]
    E --> F[Sandboxed Code Execution]
    F --> G[Result Serialization]
    G --> H[Result Delivery]
    H --> C
}}

== Related Pages ==

=== Steps ===
* [[step::Principle:n8n-io_n8n_Runner_Initialization]]
* [[step::Principle:n8n-io_n8n_Broker_Connection]]
* [[step::Principle:n8n-io_n8n_Task_Offer_Negotiation]]
* [[step::Principle:n8n-io_n8n_Security_Validation]]
* [[step::Principle:n8n-io_n8n_Subprocess_Creation]]
* [[step::Principle:n8n-io_n8n_Sandboxed_Execution]]
* [[step::Principle:n8n-io_n8n_Result_Serialization]]
* [[step::Principle:n8n-io_n8n_Result_Delivery]]

=== Related Concepts ===
* [[related::Workflow:n8n-io_n8n_Security_Validation_Pipeline]] - Security validation sub-workflow
