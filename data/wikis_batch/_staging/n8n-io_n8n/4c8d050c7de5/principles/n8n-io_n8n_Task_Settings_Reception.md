{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|n8n Python Task Runner|https://docs.n8n.io]]
|-
! Domains
| [[domain::Distributed_Systems]], [[domain::Task_Processing]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Receiving and validating task configuration data before execution in a distributed task processing system.

=== Description ===

After a task runner accepts a task from a broker, the broker transmits detailed task settings that include the executable code, input data items, and execution parameters. The task settings reception principle defines how a worker process receives this configuration payload, validates that it matches the expected task identifier and execution state, and prepares the execution environment with the provided parameters.

This principle solves the problem of coordinating distributed task execution by ensuring that:
* The worker is in the correct state to receive settings
* The settings correspond to the accepted task
* All required execution parameters are present
* The execution can proceed with validated configuration

=== Usage ===

Apply this principle when:
* Implementing distributed task processing systems
* Coordinating between task brokers and worker processes
* Validating multi-step task handoff protocols
* Ensuring state consistency in asynchronous task execution

== Theoretical Basis ==

The task settings reception follows a state machine pattern:

<pre>
State Transitions:
  IDLE -> ACCEPTED (on task offer acceptance)
  ACCEPTED -> EXECUTING (on settings reception)

Settings Reception Protocol:
  1. Verify current state == ACCEPTED
  2. Verify settings.taskId == accepted_taskId
  3. Extract code, items, and parameters
  4. Validate required fields are present
  5. Transition to EXECUTING state
  6. Trigger execution with settings

Error Conditions:
  - Settings received in wrong state -> Reject
  - Task ID mismatch -> Reject
  - Missing required fields -> Reject
</pre>

The principle ensures atomicity of the state transition and prevents race conditions where settings might arrive before acceptance or after execution has already begun.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_TaskRunner_handle_task_settings]]
