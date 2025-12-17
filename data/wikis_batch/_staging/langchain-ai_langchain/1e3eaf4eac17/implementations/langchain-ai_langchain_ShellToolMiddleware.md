{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai/langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::Agent Middleware]], [[domain::Shell Execution]], [[domain::Security]], [[domain::Code Execution]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==
ShellToolMiddleware provides agents with a persistent shell tool that enables sequential command execution with configurable execution policies, output limits, timeout controls, and PII redaction for safe and controlled shell access.

=== Description ===
This middleware registers a long-lived shell session as a tool that agents can invoke to execute commands. Unlike one-shot command execution, the shell maintains state across invocations (e.g., environment variables, working directory, background processes).

The middleware supports three execution policies:
* '''HostExecutionPolicy''': Direct execution on the host system (trusted environments)
* '''CodexSandboxExecutionPolicy''': Execution within Codex CLI sandbox (syscall/filesystem restrictions)
* '''DockerExecutionPolicy''': Execution in isolated Docker container (hardened isolation)

Key features include:
* Persistent shell sessions with state preservation
* Configurable command and output timeouts
* Output truncation by line count or byte size
* Startup and shutdown command sequences
* PII redaction for command output
* Session restart capability
* Automatic resource cleanup
* Concurrent stdout/stderr capture with proper ordering

=== Usage ===
Use this middleware when you need agents to:
* Execute multi-step workflows requiring persistent state
* Run system commands in controlled environments
* Manage files, processes, or system configurations
* Perform development tasks (compilation, testing, deployment)
* Debug or troubleshoot systems interactively

Choose execution policy based on trust level and isolation requirements.

== Code Reference ==
'''Source location:''' `/tmp/praxium_repo_wjjl6pl8/libs/langchain_v1/langchain/agents/middleware/shell_tool.py`

'''Signature:'''
<syntaxhighlight lang="python">
class ShellToolMiddleware(AgentMiddleware[ShellToolState, Any]):
    def __init__(
        self,
        workspace_root: str | Path | None = None,
        *,
        startup_commands: tuple[str, ...] | list[str] | str | None = None,
        shutdown_commands: tuple[str, ...] | list[str] | str | None = None,
        execution_policy: BaseExecutionPolicy | None = None,
        redaction_rules: tuple[RedactionRule, ...] | list[RedactionRule] | None = None,
        tool_description: str | None = None,
        tool_name: str = "shell",
        shell_command: Sequence[str] | str | None = None,
        env: Mapping[str, Any] | None = None,
    ) -> None
</syntaxhighlight>

'''Import statement:'''
<syntaxhighlight lang="python">
from langchain.agents.middleware import ShellToolMiddleware
from langchain.agents.middleware.shell_tool import (
    HostExecutionPolicy,
    DockerExecutionPolicy,
    CodexSandboxExecutionPolicy,
    RedactionRule
)
</syntaxhighlight>

== I/O Contract ==

=== Initialization Parameters ===
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| workspace_root || str/Path/None || None || Base directory for shell session. If None, creates temporary directory
|-
| startup_commands || tuple/list/str/None || None || Commands executed sequentially after session starts
|-
| shutdown_commands || tuple/list/str/None || None || Commands executed before session shuts down
|-
| execution_policy || BaseExecutionPolicy/None || HostExecutionPolicy() || Policy controlling timeouts, limits, and environment
|-
| redaction_rules || tuple/list/None || None || PII redaction rules to sanitize command output
|-
| tool_description || str/None || DEFAULT_TOOL_DESCRIPTION || Custom description for the registered tool
|-
| tool_name || str || "shell" || Name for the registered shell tool
|-
| shell_command || Sequence/str/None || "/bin/bash" || Shell executable or argument sequence
|-
| env || Mapping/None || None || Environment variables for shell session (inherits parent if None)
|}

=== State Schema Extension ===
{| class="wikitable"
! Field !! Type !! Description
|-
| shell_session_resources || _SessionResources/None || Private state tracking shell session, temp directory, and policy
|}

=== Tool Input Schema ===
{| class="wikitable"
! Field !! Type !! Required !! Description
|-
| command || str || Conditional || Shell command to execute (required unless restart=True)
|-
| restart || bool || No || Whether to restart the shell session
|}

=== Tool Output ===
Returns `ToolMessage` with:
* '''Content''': Command output (stdout/stderr merged, truncation notices, exit codes)
* '''Status''': "success" (exit code 0) or "error" (non-zero exit code or timeout)
* '''Artifact''': Metadata including exit_code, timed_out, truncation flags, line/byte counts, redaction matches

=== Hook Methods ===
{| class="wikitable"
! Method !! Execution Point !! Purpose
|-
| before_agent() || Agent start || Initialize shell session, run startup commands
|-
| after_agent() || Agent completion || Run shutdown commands, cleanup resources
|}

== Usage Examples ==

=== Example 1: Basic Shell Access (Host Policy) ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.agents.middleware import ShellToolMiddleware

# Direct host access (use only in trusted environments)
agent = create_agent(
    "openai:gpt-4o",
    middleware=[
        ShellToolMiddleware(
            workspace_root="/tmp/agent_workspace"
        )
    ],
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "List files in the workspace"}]
})
# Agent can use shell tool: ls, pwd, mkdir, etc.
</syntaxhighlight>

=== Example 2: Docker Isolation Policy ===
<syntaxhighlight lang="python">
from langchain.agents.middleware.shell_tool import (
    ShellToolMiddleware,
    DockerExecutionPolicy
)

# Execute in isolated Docker container
agent = create_agent(
    "openai:gpt-4o",
    middleware=[
        ShellToolMiddleware(
            execution_policy=DockerExecutionPolicy(
                image="ubuntu:22.04",
                command_timeout=30.0,
                max_output_lines=500,
                read_only_root=True,      # Read-only root filesystem
                user="nobody",            # Run as non-root user
            )
        )
    ],
)

# Commands execute safely in container with limited privileges
</syntaxhighlight>

=== Example 3: Startup and Shutdown Commands ===
<syntaxhighlight lang="python">
# Prepare environment on start, cleanup on exit
agent = create_agent(
    "openai:gpt-4o",
    middleware=[
        ShellToolMiddleware(
            workspace_root="/tmp/build_workspace",
            startup_commands=[
                "export PATH=$PATH:/usr/local/custom/bin",
                "source /etc/profile.d/myapp.sh",
                "mkdir -p logs artifacts",
            ],
            shutdown_commands=[
                "tar -czf /tmp/logs_$(date +%s).tar.gz logs/",
                "rm -rf artifacts/*",
            ],
        )
    ],
)

# Startup commands run before agent starts
# Shutdown commands run when agent completes (success or failure)
</syntaxhighlight>

=== Example 4: Custom Environment Variables ===
<syntaxhighlight lang="python">
# Pass custom environment to shell
agent = create_agent(
    "openai:gpt-4o",
    middleware=[
        ShellToolMiddleware(
            env={
                "API_KEY": "secret-key-123",
                "DEBUG": "1",
                "LOG_LEVEL": "INFO",
                "MAX_WORKERS": 4,  # Coerced to string "4"
            }
        )
    ],
)

# Agent can access environment variables in shell commands
# Example: echo $API_KEY, python script.py (reads LOG_LEVEL)
</syntaxhighlight>

=== Example 5: Output Redaction ===
<syntaxhighlight lang="python">
from langchain.agents.middleware.shell_tool import RedactionRule

# Redact sensitive information from command output
agent = create_agent(
    "openai:gpt-4o",
    middleware=[
        ShellToolMiddleware(
            redaction_rules=[
                RedactionRule("email", strategy="redact"),
                RedactionRule("ip", strategy="hash"),
                RedactionRule("api_key", detector=r"sk-[a-zA-Z0-9]{32}", strategy="block"),
            ]
        )
    ],
)

# Command output sanitized before returning to model
# Blocks execution if API keys detected in output
</syntaxhighlight>

=== Example 6: Custom Execution Policy ===
<syntaxhighlight lang="python">
from langchain.agents.middleware.shell_tool import HostExecutionPolicy

# Customize timeouts and limits
agent = create_agent(
    "openai:gpt-4o",
    middleware=[
        ShellToolMiddleware(
            execution_policy=HostExecutionPolicy(
                command_timeout=60.0,          # 60 seconds per command
                startup_timeout=10.0,          # 10 seconds for startup commands
                termination_timeout=5.0,       # 5 seconds to gracefully terminate
                max_output_lines=1000,         # Truncate after 1000 lines
                max_output_bytes=1_000_000,    # Truncate after 1MB
            )
        )
    ],
)

# Long-running commands terminated after 60s
# Output truncated if exceeds limits (agent notified via truncation message)
</syntaxhighlight>

=== Example 7: Session Restart ===
<syntaxhighlight lang="python">
# Agent can restart shell session if needed
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "If the shell seems stuck, restart it and try again"
    }]
})

# Agent can invoke tool with restart=True:
# {"command": null, "restart": true}
# Session restarted, startup commands re-run
</syntaxhighlight>

=== Example 8: Development Workflow ===
<syntaxhighlight lang="python">
# Configure for development tasks
agent = create_agent(
    "openai:gpt-4o",
    middleware=[
        ShellToolMiddleware(
            workspace_root="./project",
            startup_commands=[
                "source venv/bin/activate",
                "export PYTHONPATH=$PWD/src",
            ],
            shutdown_commands=[
                "deactivate",
            ],
            execution_policy=HostExecutionPolicy(
                command_timeout=300.0,  # 5 minutes for long builds
                max_output_lines=5000,
            )
        )
    ],
)

# Agent can run development commands:
# - python -m pytest tests/
# - mypy src/
# - black src/ tests/
# All commands share activated virtualenv
</syntaxhighlight>

=== Example 9: Codex Sandbox Policy (CLI Available) ===
<syntaxhighlight lang="python">
from langchain.agents.middleware.shell_tool import CodexSandboxExecutionPolicy

# Use Codex CLI sandbox for additional restrictions
agent = create_agent(
    "openai:gpt-4o",
    middleware=[
        ShellToolMiddleware(
            execution_policy=CodexSandboxExecutionPolicy(
                command_timeout=45.0,
                max_output_lines=750,
            )
        )
    ],
)

# Commands run with Codex sandbox restrictions
# (requires Codex CLI installed and available)
</syntaxhighlight>

== Implementation Details ==

=== Shell Session Management ===
The middleware maintains a persistent `ShellSession`:
* Spawns subprocess with stdin/stdout/stderr pipes
* Reader threads continuously enqueue output from stdout/stderr
* Commands written to stdin with unique markers for completion detection
* Output collected until marker detected or timeout reached
* Session automatically restarted on timeout

=== Command Execution Flow ===
# Write command + completion marker to stdin
# Collect output from queue (stdout and stderr merged)
# Detect completion marker on stdout (format: `__LC_SHELL_DONE__<uuid> <exit_code>`)
# Drain remaining stderr (concurrent stderr may arrive after marker)
# Apply redaction rules to output
# Format result with truncation notices and exit code

=== Output Truncation ===
When output exceeds limits:
* Lines/bytes continue to be counted but not collected
* Truncation notice appended to output
* Model informed of actual vs. truncated counts

Example truncation message:
```
... Output truncated at 1000 lines (observed 2347).
... Output truncated at 1000000 bytes (observed 3456789).
```

=== Resource Cleanup ===
Cleanup handled via `weakref.finalize`:
* Automatically triggered when middleware or session garbage collected
* Runs shutdown commands
* Terminates shell process (graceful exit, then SIGKILL)
* Removes temporary workspace directory (if created)

=== Stderr Handling ===
Stderr prefixed with `[stderr]` in output:
```
normal stdout output
[stderr] error message here
more stdout
```

=== Session Restart Logic ===
On timeout or explicit restart:
# Kill existing process (SIGKILL)
# Clear output queue
# Spawn new process
# Re-run startup commands

=== Tool Call Interception ===
The shell tool is registered as a native tool (not intercepted):
* Uses `@tool` decorator with `ToolRuntime` injection
* Accesses state directly to retrieve session resources
* No `wrap_tool_call` needed (direct execution)

=== Execution Policies ===

'''HostExecutionPolicy:'''
* Direct subprocess execution on host
* Inherits full host environment
* No isolation (use only in trusted environments)
* Fastest execution

'''DockerExecutionPolicy:'''
* Spawns Docker container per agent run
* Supports custom images, user mapping, read-only root
* Network isolation options
* Volume mounting for workspace
* Overhead: container start/stop time

'''CodexSandboxExecutionPolicy:'''
* Uses Codex CLI sandbox (if available)
* Syscall filtering and filesystem restrictions
* Moderate isolation
* Requires Codex CLI installed

=== State Management ===
State field `shell_session_resources`:
* Type: `_SessionResources` (contains session, tempdir, policy)
* Marked as `UntrackedValue` (not serialized)
* Marked as `PrivateStateAttr` (not exposed to tools/model)
* Enables resumability (checks existing resources before creating new)

=== Error Handling ===
* Startup command failures: Raise `RuntimeError`, cleanup resources
* Command timeouts: Restart session, return error ToolMessage
* Redaction blocks: Return error ToolMessage with PII type
* Invalid tool input: Raise `ToolException`

== Related Pages ==
* [[langchain-ai_langchain_AgentMiddleware|AgentMiddleware]] - Base middleware class
* [[langchain-ai_langchain_PIIMiddleware|PIIMiddleware]] - PII detection patterns
* [[langchain-ai_langchain_RedactionRule|RedactionRule]] - Output sanitization
* [[Execution Policies Guide]] - Choosing security policies
* [[Agent Tool Security]] - Best practices for tool safety
