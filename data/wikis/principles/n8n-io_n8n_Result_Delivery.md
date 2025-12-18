# Principle: Result Delivery

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Task_Execution]], [[domain::Distributed_Systems]], [[domain::Error_Handling]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Principle for orchestrating complete task lifecycle from settings receipt through execution to result or error delivery back to the broker.

=== Description ===

Result Delivery is the top-level orchestration of task execution:

1. **Validation**: Run security analysis on task code
2. **Process Management**: Create and start subprocess with proper cleanup
3. **Execution Monitoring**: Wait for result with timeout handling
4. **Print Forwarding**: Send captured print() calls to browser console via RPC
5. **Success Handling**: Package and send `RunnerTaskDone` with result data
6. **Error Handling**: Package and send `RunnerTaskError` with error details
7. **Cleanup**: Remove task from running_tasks, reset idle timer

This orchestration ensures:
- **Complete Lifecycle**: All phases properly sequenced
- **Resource Cleanup**: Tasks always removed from tracking
- **Error Propagation**: Detailed errors sent to broker
- **User Feedback**: Print statements forwarded to browser

=== Usage ===

Apply this principle when:
- Implementing task worker orchestration logic
- Building systems with multi-phase task execution
- Designing error handling for distributed task systems
- Creating execution pipelines with cleanup guarantees

== Theoretical Basis ==

Result delivery follows an **Orchestration with Cleanup** pattern:

<syntaxhighlight lang="python">
# Pseudo-code for result delivery

async def execute_task(task_id, task_settings):
    try:
        # 1. Validate code security
        analyzer.validate(task_settings.code)

        # 2. Create and run subprocess
        process, read_conn, write_conn = executor.create_process(...)
        result, print_args, size = await executor.execute_process(...)

        # 3. Forward print statements
        for args in print_args:
            await send_rpc("console.log", args)

        # 4. Send success response
        await send(RunnerTaskDone(task_id, {"result": result}))

    except TaskCancelledError as e:
        await send(RunnerTaskError(task_id, {"message": str(e)}))

    except SyntaxError as e:
        await send(RunnerTaskError(task_id, {"message": str(e)}))

    except Exception as e:
        await send(RunnerTaskError(task_id, {
            "message": getattr(e, "message", str(e)),
            "description": getattr(e, "description", "")
        }))

    finally:
        # 5. Always cleanup
        running_tasks.pop(task_id, None)
        reset_idle_timer()
</syntaxhighlight>

Error types handled:
- **TaskCancelledError**: Task was cancelled by broker
- **SyntaxError**: Code failed to parse
- **SecurityViolationError**: Code failed security validation
- **TaskTimeoutError**: Execution exceeded timeout
- **TaskRuntimeError**: User code raised exception

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_TaskRunner_execute_task]]
