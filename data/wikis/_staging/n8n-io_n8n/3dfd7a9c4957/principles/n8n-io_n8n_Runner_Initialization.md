# Principle: Runner Initialization

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
* [[source::Doc|n8n Docs|https://docs.n8n.io]]
|-
! Domains
| [[domain::Task_Execution]], [[domain::Distributed_Systems]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Principle for initializing a task runner service with configuration, establishing the necessary components for secure, distributed Python code execution.

=== Description ===

Runner Initialization is the first step in setting up a Python task execution service. It establishes the runtime environment by:

1. **Loading Configuration**: Reading environment variables to configure broker URI, concurrency limits, timeouts, and security policies
2. **Creating Core Components**: Instantiating the TaskExecutor for subprocess management, TaskAnalyzer for security validation, and WebSocket connection handler
3. **Establishing Identity**: Generating a unique runner ID for broker communication and task tracking
4. **Setting Security Context**: Initializing SecurityConfig with allowlists and denylists for module imports and builtins

This principle ensures that all subsequent task operations have a properly configured environment with appropriate security constraints.

=== Usage ===

Apply this principle when:
- Setting up a new Python code execution service
- Implementing a distributed task worker that connects to a central broker
- Building a sandbox environment that requires pre-execution security configuration
- Creating an isolated execution context with configurable resource limits

== Theoretical Basis ==

The initialization pattern follows the **Dependency Injection** principle:

<syntaxhighlight lang="python">
# Pseudo-code for runner initialization
runner = TaskRunner(config)
  # 1. Generate unique identity
  runner.id = generate_unique_id()

  # 2. Create security configuration
  security_config = SecurityConfig(
    stdlib_allow=config.stdlib_allow,
    external_allow=config.external_allow,
    builtins_deny=config.builtins_deny
  )

  # 3. Initialize executor with security context
  runner.executor = TaskExecutor()
  runner.analyzer = TaskAnalyzer(security_config)

  # 4. Prepare WebSocket connection parameters
  runner.websocket_url = build_url(config.broker_uri, runner.id)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_TaskRunner_init]]
