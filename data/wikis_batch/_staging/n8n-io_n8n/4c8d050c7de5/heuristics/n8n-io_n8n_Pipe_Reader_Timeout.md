# Heuristic: Pipe Reader Timeout

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n Task Runner Python|https://github.com/n8n-io/n8n/tree/master/packages/@n8n/task-runner-python]]
|-
! Domains
| [[domain::IPC]], [[domain::Optimization]], [[domain::Reliability]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

Dynamic pipe reader timeout calculated from max payload size: `(max_payload * 0.1) / 100MB_per_sec + 2.0s buffer`, preventing hung threads while accommodating large payloads.

=== Description ===

The task executor uses a background thread (PipeReader) to read subprocess results over an IPC pipe. The timeout for this thread is dynamically calculated based on the configured maximum payload size, ensuring that large payloads have sufficient time to be read while preventing indefinite hangs on stuck subprocesses.

=== Usage ===

This heuristic applies automatically when configuring the **TaskRunnerConfig** from environment variables. Understanding this timeout calculation is critical when:

- Debugging "Pipe reader thread did not finish reading" warnings
- Configuring runners for large payload workloads
- Troubleshooting task failures due to partial data reads

== The Insight (Rule of Thumb) ==

* **Formula:** `timeout = (max_payload * TYPICAL_PAYLOAD_RATIO / PARSE_THROUGHPUT) + SAFETY_BUFFER`
* **Constants:**
  - `TYPICAL_PAYLOAD_RATIO = 0.1` (assume 10% of max is typical)
  - `PARSE_THROUGHPUT_BYTES_PER_SEC = 100_000_000` (100 MB/s)
  - `PIPE_READER_JOIN_TIMEOUT_SAFETY_BUFFER = 2.0` seconds
* **Default Timeout:** ~3 seconds for 1 GiB max payload
* **Trade-off:** Longer timeouts prevent data loss but delay error detection; shorter timeouts may truncate valid large payloads

== Reasoning ==

The timeout calculation is designed to:

1. **Scale with Payload Size:** Larger `max_payload_size` automatically gets more read time
2. **Assume Typical Case:** Using 10% of max avoids over-provisioning for edge cases
3. **Conservative Throughput:** 100 MB/s is achievable for JSON parsing on most systems
4. **Safety Buffer:** 2-second buffer handles process cleanup and edge cases

For the default 1 GiB max payload:
```
typical_payload = 1 GiB * 0.1 = 100 MB
read_time = 100 MB / 100 MB/s = 1.0 second
total_timeout = 1.0 + 2.0 = 3.0 seconds
```

When the timeout is triggered, the pipe is forcibly closed which may cause the task to fail if data was not fully read.

== Code Evidence ==

Constants from `constants.py:52-55`:
<syntaxhighlight lang="python">
# Pipe reader join timeout
TYPICAL_PAYLOAD_RATIO = 0.1  # assume typical size is 10% of max payload
PARSE_THROUGHPUT_BYTES_PER_SEC = 100_000_000  # 100 MB/s
PIPE_READER_JOIN_TIMEOUT_SAFETY_BUFFER = 2.0  # seconds
</syntaxhighlight>

Timeout calculation from `task_runner_config.py:105-109`:
<syntaxhighlight lang="python">
# Calculate pipe reader timeout based on configured max payload size (3s for default 1 GiB)
typical_payload = max_payload_size * TYPICAL_PAYLOAD_RATIO
pipe_reader_timeout = (
    typical_payload / PARSE_THROUGHPUT_BYTES_PER_SEC
) + PIPE_READER_JOIN_TIMEOUT_SAFETY_BUFFER
</syntaxhighlight>

Warning message from `constants.py:109-113`:
<syntaxhighlight lang="python">
LOG_PIPE_READER_TIMEOUT_TRIGGERED = (
    "Pipe reader thread did not finish reading within {timeout}s. "
    "Closing pipe to unblock. Task may fail if data was not fully read. "
    "For large payloads, increase N8N_RUNNERS_MAX_PAYLOAD to scale timeout."
)
</syntaxhighlight>

Warning trigger from `task_executor.py:131`:
<syntaxhighlight lang="python">
logger.warning(
    LOG_PIPE_READER_TIMEOUT_TRIGGERED.format(timeout=pipe_reader_timeout)
)
</syntaxhighlight>

== Related Pages ==

* [[uses_heuristic::Implementation:n8n-io_n8n_TaskExecutor_execute_process]]
* [[uses_heuristic::Principle:n8n-io_n8n_Result_Collection]]
