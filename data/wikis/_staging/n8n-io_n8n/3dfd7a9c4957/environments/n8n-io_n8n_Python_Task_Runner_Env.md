# Environment: Python Task Runner Environment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n-io/n8n|https://github.com/n8n-io/n8n]]
* [[source::Doc|pyproject.toml|packages/@n8n/task-runner-python/pyproject.toml]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Task_Execution]], [[domain::Security]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==
Linux/macOS Python 3.13+ environment with websockets for running sandboxed Python code tasks in n8n workflows.

=== Description ===
This environment provides the runtime context for the n8n Python Task Runner, a secure subprocess-based code execution system. It uses Python's `forkserver` multiprocessing context for process isolation, WebSocket communication with the n8n task broker, and comprehensive security controls via AST analysis and runtime import validation. The environment explicitly **does not support Windows** due to the forkserver requirement.

=== Usage ===
Use this environment when running the **Python Task Execution** workflow for executing user-provided Python code in n8n workflows. It is required for the `TaskRunner`, `TaskExecutor`, and `TaskAnalyzer` implementations.

== System Requirements ==

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux or macOS || Windows is explicitly not supported (forkserver unavailable)
|-
| Python || >= 3.13 || Uses `sys.stdlib_module_names` from Python 3.10+, type syntax from 3.13
|-
| Network || WebSocket connectivity to broker || Default: `http://127.0.0.1:5679/runners/_ws`
|}

== Dependencies ==

=== System Packages ===
* Python 3.13+

=== Python Packages ===
* `websockets` >= 15.0.1 - WebSocket client for broker communication

=== Optional Python Packages ===
* `sentry-sdk` >= 2.35.2 - For error tracking (optional)

== Quick Install ==

<syntaxhighlight lang="bash">
# Requires Python 3.13+
pip install websockets>=15.0.1

# Optional: For error tracking
pip install sentry-sdk>=2.35.2
</syntaxhighlight>

== Credentials ==

The following environment variables must be configured:

=== Required ===
* `N8N_RUNNERS_GRANT_TOKEN`: Authentication token for broker connection (REQUIRED)

=== Optional ===
* `N8N_RUNNERS_TASK_BROKER_URI`: Broker URI (default: `http://127.0.0.1:5679`)
* `N8N_RUNNERS_MAX_CONCURRENCY`: Max concurrent tasks (default: 5)
* `N8N_RUNNERS_MAX_PAYLOAD`: Max payload size in bytes (default: 1 GiB)
* `N8N_RUNNERS_TASK_TIMEOUT`: Task execution timeout in seconds (default: 60)
* `N8N_RUNNERS_AUTO_SHUTDOWN_TIMEOUT`: Idle shutdown timeout (default: 0/disabled)
* `N8N_RUNNERS_GRACEFUL_SHUTDOWN_TIMEOUT`: Graceful shutdown timeout (default: 10s)

=== Security Configuration ===
* `N8N_RUNNERS_STDLIB_ALLOW`: Comma-separated stdlib modules to allow (default: none, use `*` for all)
* `N8N_RUNNERS_EXTERNAL_ALLOW`: Comma-separated external packages to allow (default: none, use `*` for all)
* `N8N_RUNNERS_BUILTINS_DENY`: Comma-separated builtins to deny (default: eval,exec,compile,open,input,breakpoint,getattr,object,type,vars,setattr,delattr,hasattr,dir,memoryview,__build_class__,globals,locals,license,help,credits,copyright)
* `N8N_BLOCK_RUNNER_ENV_ACCESS`: Block access to environment variables (default: true)

=== Optional Health Check ===
* `N8N_RUNNERS_HEALTH_CHECK_SERVER_ENABLED`: Enable health check server
* `N8N_RUNNERS_HEALTH_CHECK_SERVER_HOST`: Health check host (default: 127.0.0.1)
* `N8N_RUNNERS_HEALTH_CHECK_SERVER_PORT`: Health check port (default: 5681)

=== Optional Sentry ===
* `N8N_SENTRY_DSN`: Sentry DSN for error tracking
* `N8N_VERSION`: n8n version tag for Sentry
* `ENVIRONMENT`: Environment name for Sentry
* `DEPLOYMENT_NAME`: Deployment name for Sentry

== Code Evidence ==

Platform check from `main.py:67-70`:
<syntaxhighlight lang="python">
if platform.system() == "Windows":
    print(ERROR_WINDOWS_NOT_SUPPORTED, file=sys.stderr)
    sys.exit(1)
</syntaxhighlight>

Required token validation from `task_runner_config.py:71-75`:
<syntaxhighlight lang="python">
grant_token = read_str_env(ENV_GRANT_TOKEN, "")
if not grant_token:
    raise ConfigurationError(
        "Environment variable N8N_RUNNERS_GRANT_TOKEN is required"
    )
</syntaxhighlight>

Forkserver context from `task_executor.py:47`:
<syntaxhighlight lang="python">
MULTIPROCESSING_CONTEXT = multiprocessing.get_context("forkserver")
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `Error: This task runner is not supported on Windows` || Running on Windows OS || Use Linux or macOS
|-
|| `Environment variable N8N_RUNNERS_GRANT_TOKEN is required` || Missing authentication token || Set `N8N_RUNNERS_GRANT_TOKEN` environment variable
|-
|| `Task timeout must be positive` || Invalid timeout configuration || Set `N8N_RUNNERS_TASK_TIMEOUT` to a positive integer
|-
|| `Max payload size exceeds pipe message limit` || Payload size too large || Reduce `N8N_RUNNERS_MAX_PAYLOAD` to below ~4 GiB
|-
|| `Failed to connect to broker: ... - retrying...` || Broker not reachable || Verify broker URI and ensure broker is running
|}

== Compatibility Notes ==

* **Windows:** Explicitly not supported. The task runner uses Python's `forkserver` multiprocessing context which is unavailable on Windows.
* **macOS:** Fully supported. Uses `forkserver` context.
* **Linux:** Fully supported (primary target platform).
* **Docker:** Recommended deployment method for production.

== Related Pages ==

* [[requires_env::Implementation:n8n-io_n8n_TaskRunner_init]]
* [[requires_env::Implementation:n8n-io_n8n_TaskRunner_start]]
* [[requires_env::Implementation:n8n-io_n8n_TaskRunner_send_offers]]
* [[requires_env::Implementation:n8n-io_n8n_TaskRunner_execute_task]]
* [[requires_env::Implementation:n8n-io_n8n_TaskAnalyzer_validate]]
* [[requires_env::Implementation:n8n-io_n8n_TaskExecutor_create_process]]
* [[requires_env::Implementation:n8n-io_n8n_SecurityConfig]]
* [[requires_env::Implementation:n8n-io_n8n_TaskAnalyzer_cache]]
