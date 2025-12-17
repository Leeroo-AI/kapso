# Environment: Python Task Runner

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n Task Runner Python|https://github.com/n8n-io/n8n/tree/master/packages/@n8n/task-runner-python]]
* [[source::Doc|pyproject.toml|packages/@n8n/task-runner-python/pyproject.toml]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Workflow_Automation]], [[domain::Security]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

Unix-like environment with Python 3.13+, websockets, and forkserver multiprocessing for secure Python code execution in n8n workflows.

=== Description ===

This environment provides the runtime context for the n8n Python Task Runner, which executes user-provided Python code in isolated subprocess contexts. The environment requires a Unix-like operating system (Linux or macOS) due to its use of the `forkserver` multiprocessing start method, which is not available on Windows. It includes security mechanisms for sandboxing code execution including AST-based static analysis and runtime import validation.

=== Usage ===

Use this environment for running the **n8n Python Task Runner** component, which handles Python code execution within n8n workflows. It is required for all implementations in the Python Task Execution and Security Validation workflows.

== System Requirements ==

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Unix-like (Linux, macOS) || Windows explicitly NOT supported
|-
| Python || >= 3.13 || Required for type annotations and multiprocessing features
|-
| Multiprocessing || forkserver context || Required for subprocess isolation
|-
| Network || WebSocket connectivity || For broker communication (default: ws://127.0.0.1:5679)
|}

== Dependencies ==

=== System Packages ===

* Python >= 3.13

=== Python Packages ===

* `websockets` >= 15.0.1 (WebSocket communication with broker)

=== Optional Dependencies ===

* `sentry-sdk` >= 2.35.2 (Error tracking, install with `uv sync --all-extras`)
* `urllib3` >= 2.6.0 (Constrained for security)

=== Development Dependencies ===

* `ruff` >= 0.12.8
* `pytest` >= 8.0.0
* `pytest-cov` >= 5.0.0
* `pytest-asyncio` >= 0.24.0
* `aiohttp` >= 3.10.0

== Credentials ==

The following environment variables must be set:

* `N8N_RUNNERS_GRANT_TOKEN`: **Required.** Authentication token for broker registration.
* `N8N_RUNNERS_TASK_BROKER_URI`: Task broker WebSocket URI (default: `http://127.0.0.1:5679`)
* `N8N_RUNNERS_MAX_CONCURRENCY`: Maximum concurrent tasks (default: 5)
* `N8N_RUNNERS_MAX_PAYLOAD`: Maximum payload size in bytes (default: 1 GiB)
* `N8N_RUNNERS_TASK_TIMEOUT`: Task timeout in seconds (default: 60)
* `N8N_RUNNERS_AUTO_SHUTDOWN_TIMEOUT`: Auto-shutdown timeout (default: 0, disabled)
* `N8N_RUNNERS_GRACEFUL_SHUTDOWN_TIMEOUT`: Graceful shutdown timeout (default: 10s)
* `N8N_RUNNERS_STDLIB_ALLOW`: Comma-separated list of allowed stdlib modules
* `N8N_RUNNERS_EXTERNAL_ALLOW`: Comma-separated list of allowed external packages
* `N8N_RUNNERS_BUILTINS_DENY`: Comma-separated list of denied builtins
* `N8N_RUNNERS_HEALTH_CHECK_SERVER_ENABLED`: Enable health check endpoint
* `N8N_RUNNERS_HEALTH_CHECK_SERVER_HOST`: Health check host (default: 127.0.0.1)
* `N8N_RUNNERS_HEALTH_CHECK_SERVER_PORT`: Health check port (default: 5681)
* `N8N_SENTRY_DSN`: Sentry DSN for error tracking (optional)

**Note:** All environment variables support the `_FILE` suffix for Docker secrets (e.g., `N8N_RUNNERS_GRANT_TOKEN_FILE`).

== Quick Install ==

<syntaxhighlight lang="bash">
# Requires Python 3.13+
pip install websockets>=15.0.1

# Optional: Sentry integration
pip install sentry-sdk>=2.35.2
</syntaxhighlight>

== Code Evidence ==

Platform check from `main.py:67-70`:
<syntaxhighlight lang="python">
if __name__ == "__main__":
    if platform.system() == "Windows":
        print(ERROR_WINDOWS_NOT_SUPPORTED, file=sys.stderr)
        sys.exit(1)
</syntaxhighlight>

Error message from `constants.py:190-193`:
<syntaxhighlight lang="python">
ERROR_WINDOWS_NOT_SUPPORTED = (
    "Error: This task runner is not supported on Windows. "
    "Please use a Unix-like system (Linux or macOS)."
)
</syntaxhighlight>

Required grant token validation from `task_runner_config.py:70-75`:
<syntaxhighlight lang="python">
grant_token = read_str_env(ENV_GRANT_TOKEN, "")
if not grant_token:
    raise ConfigurationError(
        "Environment variable N8N_RUNNERS_GRANT_TOKEN is required"
    )
</syntaxhighlight>

Environment variable file support from `env.py:5-19`:
<syntaxhighlight lang="python">
def read_env(env_name: str) -> str | None:
    if env_name in os.environ:
        return os.environ[env_name]

    file_path_key = f"{env_name}_FILE"
    if file_path_key in os.environ:
        file_path = os.environ[file_path_key]
        try:
            return Path(file_path).read_text(encoding="utf-8").strip()
        except (OSError, IOError) as e:
            raise ValueError(
                f"Failed to read {env_name}_FILE from file {file_path}: {e}"
            )
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `Error: This task runner is not supported on Windows` || Running on Windows || Use Linux or macOS
|-
|| `Environment variable N8N_RUNNERS_GRANT_TOKEN is required` || Missing grant token || Set `N8N_RUNNERS_GRANT_TOKEN` environment variable
|-
|| `Task timeout must be positive` || Invalid timeout config || Set `N8N_RUNNERS_TASK_TIMEOUT` to a positive integer
|-
|| `Max payload size exceeds pipe message limit` || Payload too large || Reduce `N8N_RUNNERS_MAX_PAYLOAD` below ~4 GiB
|-
|| `Wildcard '*' in {list_name} must be used alone` || Invalid allowlist config || Use `*` alone or specify individual modules
|-
|| `Failed to read {env}_FILE from file` || Missing or unreadable secrets file || Verify file path and permissions
|}

== Compatibility Notes ==

* **Windows:** Not supported. The task runner uses the `forkserver` multiprocessing start method which is only available on Unix-like systems.
* **Docker:** Supports Docker secrets via `_FILE` suffix on all environment variables.
* **Sentry:** Optional integration requires `sentry-sdk` package. Install with `uv sync --all-extras`.

== Related Pages ==

* [[requires_env::Implementation:n8n-io_n8n_TaskRunner_start]]
* [[requires_env::Implementation:n8n-io_n8n_TaskRunner_send_offers]]
* [[requires_env::Implementation:n8n-io_n8n_TaskRunner_handle_task_offer_accept]]
* [[requires_env::Implementation:n8n-io_n8n_TaskRunner_handle_task_settings]]
* [[requires_env::Implementation:n8n-io_n8n_TaskRunner_execute_task]]
* [[requires_env::Implementation:n8n-io_n8n_TaskAnalyzer_validate]]
* [[requires_env::Implementation:n8n-io_n8n_TaskAnalyzer_cache]]
* [[requires_env::Implementation:n8n-io_n8n_TaskExecutor_create_process]]
* [[requires_env::Implementation:n8n-io_n8n_TaskExecutor_all_items]]
* [[requires_env::Implementation:n8n-io_n8n_TaskExecutor_execute_process]]
* [[requires_env::Implementation:n8n-io_n8n_TaskExecutor_filter_builtins]]
* [[requires_env::Implementation:n8n-io_n8n_TaskExecutor_create_safe_import]]
* [[requires_env::Implementation:n8n-io_n8n_SecurityValidator_visit_Import]]
* [[requires_env::Implementation:n8n-io_n8n_SecurityValidator_visit_Attribute]]
* [[requires_env::Implementation:n8n-io_n8n_TaskAnalyzer_raise_security_error]]
* [[requires_env::Implementation:n8n-io_n8n_ast_parse]]
