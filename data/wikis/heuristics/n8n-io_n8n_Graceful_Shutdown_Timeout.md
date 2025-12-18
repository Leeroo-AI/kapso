# Heuristic: Graceful Shutdown with Timeout

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n-io/n8n|https://github.com/n8n-io/n8n]]
* [[source::Code|shutdown.py|packages/@n8n/task-runner-python/src/shutdown.py]]
* [[source::Code|task_runner.py|packages/@n8n/task-runner-python/src/task_runner.py]]
|-
! Domains
| [[domain::Reliability]], [[domain::Process_Management]], [[domain::Operations]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==
Wait for running tasks to complete during shutdown, but enforce a timeout (default 10s) to prevent indefinite hangs.

=== Description ===
When the Python Task Runner receives a shutdown signal, it should attempt to complete running tasks gracefully before exiting. However, waiting indefinitely could cause the process to hang if a task is stuck. This heuristic implements a two-phase shutdown: wait for tasks up to a timeout, then force-terminate remaining tasks.

=== Usage ===
Configure via `N8N_RUNNERS_GRACEFUL_SHUTDOWN_TIMEOUT` (default: 10 seconds). Set higher for long-running tasks, lower for faster restart cycles.

== The Insight (Rule of Thumb) ==

* **Action:** Implement two-phase shutdown: graceful wait → force terminate
* **Value:**
  * `DEFAULT_SHUTDOWN_TIMEOUT = 10` seconds for graceful phase
  * 1 second grace period for individual process termination
* **Trade-off:** Task completion vs. shutdown speed
* **Safety:** SIGTERM first, then SIGKILL if process doesn't respond

== Reasoning ==

1. **Data integrity:** Completing tasks ensures results are delivered to the broker

2. **Bounded wait:** 10s default prevents indefinite hangs during deploys/restarts

3. **Escalating force:** SIGTERM allows cleanup; SIGKILL guarantees termination

4. **Logging visibility:** Warnings logged when force-terminating to aid debugging

5. **Configurable:** Different deployments may need different timeout values

== Code Evidence ==

From `constants.py:32`:

<syntaxhighlight lang="python">
DEFAULT_SHUTDOWN_TIMEOUT = 10  # seconds
</syntaxhighlight>

From `task_runner.py:178-209`:

<syntaxhighlight lang="python">
async def wait_for_running_tasks_to_finish(self):
    """Wait for all running tasks to complete, terminating them if necessary."""
    timeout = self.config.graceful_shutdown_timeout
    self.logger.info(
        f"Waiting for {self.running_tasks_count} tasks to complete (timeout: {timeout}s)..."
    )

    start_time = time.time()
    while self.running_tasks and (time.time() - start_time) < timeout:
        await asyncio.sleep(0.1)

    if self.running_tasks:
        self.logger.warning(
            f"Timeout reached - still {self.running_tasks_count} tasks running. "
            f"Terminating remaining tasks..."
        )

    # Stop the websocket listener so it can't accept new tasks while stopping processes
    await self._stop_websocket_listener()

    self.logger.warning(f"Terminating {self.running_tasks_count} tasks...")

    for task_state in self.running_tasks.copy().values():
        TaskExecutor.stop_process(task_state.process)
        try:
            await asyncio.wait_for(task_state.completion_event.wait(), timeout=1)
        except asyncio.TimeoutError:
            pass

        self.running_tasks.pop(task_state.task_id, None)

    self.logger.warning("Terminated tasks")
</syntaxhighlight>

Process termination escalation from `task_executor.py:167-183`:

<syntaxhighlight lang="python">
@staticmethod
def stop_process(process: ForkServerProcess | None):
    """Stop a running subprocess, gracefully else force-killing."""

    if process is None or not process.is_alive():
        return

    try:
        process.terminate()  # SIGTERM
        process.join(timeout=1)  # 1s grace period

        if process.is_alive():
            process.kill()  # SIGKILL
            process.join()
    except (ProcessLookupError, ConnectionError, BrokenPipeError):
        # subprocess is dead or unreachable
        pass
</syntaxhighlight>

== Related Pages ==

* [[uses_heuristic::Implementation:n8n-io_n8n_TaskRunner_execute_task]]
* [[uses_heuristic::Workflow:n8n-io_n8n_Python_Task_Execution]]
* [[uses_heuristic::Principle:n8n-io_n8n_Result_Delivery]]
