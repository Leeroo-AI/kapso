# Heuristic: Pipe Timeout Scaling

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n-io/n8n|https://github.com/n8n-io/n8n]]
* [[source::Code|task_runner_config.py|packages/@n8n/task-runner-python/src/config/task_runner_config.py]]
* [[source::Code|constants.py|packages/@n8n/task-runner-python/src/constants.py]]
|-
! Domains
| [[domain::Performance]], [[domain::IPC]], [[domain::Configuration]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==
Dynamically calculate pipe reader timeout based on payload size to prevent premature timeouts with large data transfers.

=== Description ===
The Python Task Runner uses pipes for IPC between the main process and sandboxed subprocesses. For large payloads (up to 1 GiB by default), a fixed timeout would either be too short (causing failures) or too long (delaying error detection). This heuristic calculates an appropriate timeout based on the configured max payload size.

=== Usage ===
This heuristic is applied automatically during configuration loading. Adjust `N8N_RUNNERS_MAX_PAYLOAD` to change both the payload limit and the corresponding pipe reader timeout.

== The Insight (Rule of Thumb) ==

* **Action:** Calculate `pipe_reader_timeout = (typical_payload / throughput) + safety_buffer`
* **Value:**
  * `TYPICAL_PAYLOAD_RATIO = 0.1` (assume typical is 10% of max)
  * `PARSE_THROUGHPUT_BYTES_PER_SEC = 100_000_000` (100 MB/s)
  * `PIPE_READER_JOIN_TIMEOUT_SAFETY_BUFFER = 2.0` seconds
* **Trade-off:** Longer timeout for large payloads, but faster error detection for small payloads
* **Example:** For 1 GiB max payload → typical ~100 MB → ~3s timeout (1s + 2s buffer)

== Reasoning ==

1. **Fixed timeouts fail at scale:** A 5s timeout works for small payloads but fails when transferring 100MB of data
2. **Conservative estimate:** Using 10% of max payload as "typical" prevents overly long timeouts for normal usage
3. **Safety buffer:** The 2s buffer accounts for JSON serialization overhead and system variability
4. **User guidance:** If timeout is hit, the warning message directs users to increase `N8N_RUNNERS_MAX_PAYLOAD`

The formula:
```
timeout = (max_payload * 0.1 / 100MB/s) + 2s
```

For default 1 GiB:
```
timeout = (1073741824 * 0.1 / 100000000) + 2 = 1.07 + 2 ≈ 3s
```

== Code Evidence ==

From `task_runner_config.py:105-109`:

<syntaxhighlight lang="python">
# Calculate pipe reader timeout based on configured max payload size (3s for default 1 GiB)
typical_payload = max_payload_size * TYPICAL_PAYLOAD_RATIO
pipe_reader_timeout = (
    typical_payload / PARSE_THROUGHPUT_BYTES_PER_SEC
) + PIPE_READER_JOIN_TIMEOUT_SAFETY_BUFFER
</syntaxhighlight>

From `constants.py:52-55`:

<syntaxhighlight lang="python">
# Pipe reader join timeout
TYPICAL_PAYLOAD_RATIO = 0.1  # assume typical size is 10% of max payload
PARSE_THROUGHPUT_BYTES_PER_SEC = 100_000_000  # 100 MB/s
PIPE_READER_JOIN_TIMEOUT_SAFETY_BUFFER = 2.0  # seconds
</syntaxhighlight>

Warning message from `constants.py:109-112`:

<syntaxhighlight lang="python">
LOG_PIPE_READER_TIMEOUT_TRIGGERED = (
    "Pipe reader thread did not finish reading within {timeout}s. "
    "Closing pipe to unblock. Task may fail if data was not fully read. "
    "For large payloads, increase N8N_RUNNERS_MAX_PAYLOAD to scale timeout."
)
</syntaxhighlight>

== Related Pages ==

* [[uses_heuristic::Implementation:n8n-io_n8n_TaskExecutor_put_result]]
* [[uses_heuristic::Implementation:n8n-io_n8n_TaskRunner_execute_task]]
* [[uses_heuristic::Workflow:n8n-io_n8n_Python_Task_Execution]]
