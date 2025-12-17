{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|n8n Python Task Runner|https://docs.n8n.io]]
|-
! Domains
| [[domain::Task_Execution]], [[domain::Distributed_Systems]], [[domain::Workflow_Orchestration]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

End-to-end task lifecycle orchestration from validation through execution to result delivery with comprehensive error handling and state management.

=== Description ===

Task completion orchestrates the entire execution lifecycle of a single task, coordinating multiple subsystems to ensure reliable execution and result delivery. The complete flow encompasses:

1. **Validation**: Verify task parameters, input data, and code structure
2. **Static Analysis**: Security scan of user code for dangerous patterns
3. **Process Creation**: Spawn isolated subprocess with sanitized environment
4. **Code Execution**: Run user code in sandbox with appropriate mode
5. **Result Collection**: Retrieve results via IPC with timeout enforcement
6. **Error Handling**: Catch and format errors at every stage
7. **Result Formatting**: Structure output for broker consumption
8. **Completion Notification**: Send success/failure message to broker
9. **State Cleanup**: Remove task from active set, free resources

The orchestration must handle partial failures gracefully - if any step fails, subsequent steps are skipped and an error is reported. This ensures that tasks never leave the system in an inconsistent state (e.g., marked as running but actually failed).

The completion phase provides the critical feedback loop in the distributed system: the broker learns whether the task succeeded or failed, enabling it to retry failures or propagate results downstream.

=== Usage ===

Apply this principle when building task execution systems where:
* Complex multi-stage execution requires coordination
* Each stage has different failure modes requiring specific handling
* System must maintain consistent state across execution lifecycle
* Broker requires detailed completion information (success/failure/error details)
* Resource cleanup must occur even on failure
* Execution progress needs tracking for monitoring/debugging
* Tasks must be atomic (all-or-nothing execution semantics)

== Theoretical Basis ==

Task completion implements a **state machine with error recovery** at each transition:

**State Transition Graph:**

```
OFFERED → ACCEPTED → VALIDATED → EXECUTING → COMPLETED
   ↓          ↓           ↓           ↓          ↓
   └──────────┴───────────┴───────────┴─────→ FAILED
                                               ↓
                                            CLEANUP
```

**Execution Pipeline:**

```python
def execute_task(task_data):
    task_id = task_data['taskId']

    try:
        # Stage 1: Validation
        validate_task_data(task_data)

        # Stage 2: Static Analysis
        analyzer = TaskAnalyzer(task_data['code'])
        analyzer.validate()  # Raises on security violation

        # Stage 3: Process Creation
        process = create_isolated_subprocess()

        # Stage 4: Code Execution
        result = execute_in_subprocess(
            process=process,
            code=task_data['code'],
            items=task_data['items'],
            mode=task_data['mode']
        )

        # Stage 5: Result Collection
        # (handled within execute_in_subprocess)

        # Stage 6: Result Formatting
        formatted_result = {
            'taskId': task_id,
            'status': 'success',
            'data': result['items'],
            'output': result['logs']
        }

        # Stage 7: Completion Notification
        send_completion_message(formatted_result)

        return formatted_result

    except Exception as error:
        # Stage 8: Error Handling
        error_result = {
            'taskId': task_id,
            'status': 'error',
            'error': {
                'type': type(error).__name__,
                'message': str(error),
                'stage': get_current_stage()
            }
        }
        send_completion_message(error_result)
        return error_result

    finally:
        # Stage 9: Cleanup
        remove_from_active_tasks(task_id)
        release_capacity()
        cleanup_resources()
```

**Error Classification:**

| Error Type | Stage | Recovery Action |
|------------|-------|-----------------|
| ValidationError | Validation | Return error to broker immediately |
| SecurityViolation | Static Analysis | Return security error, log violation |
| ProcessError | Subprocess Creation | Return infrastructure error, retry possible |
| ExecutionError | Code Execution | Return user code error with traceback |
| TimeoutError | Result Collection | Kill process, return timeout error |
| SerializationError | Result Formatting | Return formatting error |

**Result Message Format:**

```json
// Success
{
  "type": "task:done",
  "taskId": "task-123",
  "data": {
    "result": {"items": [...]},
    "customData": {...}
  }
}

// Failure
{
  "type": "task:error",
  "taskId": "task-123",
  "error": {
    "message": "SecurityViolation: Import 'os' not allowed",
    "stage": "static_analysis",
    "code": "SECURITY_VIOLATION"
  }
}
```

**Atomicity Guarantees:**

The task execution provides **at-most-once execution** semantics:

1. Task accepted → State = EXECUTING
2. If any stage fails → State = FAILED, error reported
3. Cleanup runs regardless of success/failure
4. Broker receives exactly one completion message
5. Task removed from active set

**Idempotency Consideration:**

Tasks should be idempotent when possible, but the runner provides:
* **Single execution per acceptance**: Task ID tracked to prevent duplicate execution
* **Completion exactly once**: Only one completion message sent per task
* **State consistency**: Active task set accurately reflects running tasks

**Resource Management:**

```python
class TaskState:
    def __init__(self, task_id):
        self.task_id = task_id
        self.start_time = time.time()
        self.process = None

    def cleanup(self):
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)

        # Free capacity for new offers
        release_capacity()

        # Remove from tracking
        del active_tasks[self.task_id]
```

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_TaskRunner_execute_task]]

=== Related Principles ===
* [[Principle:n8n-io_n8n_Task_Acceptance]] - Precedes task completion
* [[Principle:n8n-io_n8n_Static_Security_Analysis]] - Validation stage
* [[Principle:n8n-io_n8n_Subprocess_Isolation]] - Process creation stage
* [[Principle:n8n-io_n8n_Code_Execution]] - Execution stage
* [[Principle:n8n-io_n8n_Result_Collection]] - Collection stage
* [[Principle:n8n-io_n8n_WebSocket_Connection]] - Sends completion notification
* [[Principle:n8n-io_n8n_Offer_Based_Distribution]] - Frees capacity for new offers
