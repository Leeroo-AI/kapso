# Implementation: TaskRunner.__init__

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Task_Execution]], [[domain::Distributed_Systems]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Concrete constructor for initializing a TaskRunner instance with configuration, security policies, and communication components.

=== Description ===

`TaskRunner.__init__` initializes the Python task runner service with all necessary components for distributed task execution. It:

1. Generates a unique runner ID using nanoid
2. Parses the broker URI to construct the WebSocket URL
3. Initializes internal state dictionaries for offers and running tasks
4. Creates the SecurityConfig from environment-sourced configuration
5. Instantiates TaskExecutor and TaskAnalyzer components
6. Sets up idle timeout handling for auto-shutdown

=== Usage ===

Use this constructor when creating a new TaskRunner instance at service startup. The configuration is typically loaded from environment variables via `TaskRunnerConfig.from_env()`.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/task-runner-python/src/task_runner.py
* '''Lines:''' L74-110

=== Signature ===
<syntaxhighlight lang="python">
class TaskRunner:
    def __init__(
        self,
        config: TaskRunnerConfig,
    ):
        """
        Initialize the TaskRunner with configuration.

        Args:
            config: TaskRunnerConfig containing broker URI, grant token,
                    max concurrency, timeouts, and security settings.
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from src.task_runner import TaskRunner
from src.config.task_runner_config import TaskRunnerConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| config || TaskRunnerConfig || Yes || Complete configuration including broker URI, grant token, concurrency limits, timeouts, and security allowlists/denylists
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| self.runner_id || str || Unique identifier for this runner instance (nanoid)
|-
| self.websocket_connection || ClientConnection | None || WebSocket connection handle (None until start() called)
|-
| self.executor || TaskExecutor || Subprocess management component
|-
| self.analyzer || TaskAnalyzer || Code security analysis component
|-
| self.security_config || SecurityConfig || Security policy configuration
|-
| self.open_offers || dict[str, TaskOffer] || Dictionary tracking task offers
|-
| self.running_tasks || dict[str, TaskState] || Dictionary tracking running tasks
|}

== Usage Examples ==

=== Basic Initialization ===
<syntaxhighlight lang="python">
from src.task_runner import TaskRunner
from src.config.task_runner_config import TaskRunnerConfig

# Load configuration from environment
config = TaskRunnerConfig.from_env()

# Initialize runner
runner = TaskRunner(config)

# Runner is now ready to start
# runner.start() to begin broker connection
</syntaxhighlight>

=== With Custom Configuration ===
<syntaxhighlight lang="python">
from dataclasses import replace

# Modify configuration for testing
test_config = replace(
    TaskRunnerConfig.from_env(),
    max_concurrency=1,
    task_timeout=30,
    stdlib_allow={"json", "datetime"},
    external_allow=set(),
)

runner = TaskRunner(test_config)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Runner_Initialization]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Task_Runner_Env]]
