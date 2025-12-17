{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai/langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::Agent Middleware]], [[domain::Security]], [[domain::Process Isolation]], [[domain::Sandboxing]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==
Abstract base class and concrete implementations for configuring how agent shell commands are executed with varying levels of isolation and resource constraints.

=== Description ===
The execution policy system provides a pluggable architecture for controlling how persistent shell sessions are spawned in agent middleware. The module defines BaseExecutionPolicy as an abstract contract and three concrete implementations: HostExecutionPolicy (direct host execution with optional resource limits), CodexSandboxExecutionPolicy (syscall filtering via Codex CLI), and DockerExecutionPolicy (container-level isolation).

Each policy encapsulates the security guarantees, resource constraints, and environmental requirements for shell execution. Policies configure timeout limits, output constraints, and platform-specific isolation mechanisms. The spawn() method returns a subprocess.Popen object that the middleware manages for command execution.

This design allows agents to adapt their execution strategy based on trust level, deployment environment, and security requirements without changing middleware code.

=== Usage ===
Use HostExecutionPolicy for trusted environments (CI, developer workstations, pre-sandboxed containers) where commands need full host access. Use CodexSandboxExecutionPolicy when the Codex CLI is available and you want additional syscall restrictions on Linux/macOS. Use DockerExecutionPolicy for untrusted user input or when strong container isolation is required. Policies can be configured with resource limits, timeouts, and environment variables.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai/langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain_v1/langchain/agents/middleware/_execution.py libs/langchain_v1/langchain/agents/middleware/_execution.py]

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class BaseExecutionPolicy(abc.ABC):
    command_timeout: float = 30.0
    startup_timeout: float = 30.0
    termination_timeout: float = 10.0
    max_output_lines: int = 100
    max_output_bytes: int | None = None

    @abc.abstractmethod
    def spawn(
        self, *, workspace: Path, env: Mapping[str, str], command: Sequence[str]
    ) -> subprocess.Popen[str]:
        """Launch the persistent shell process."""


@dataclass
class HostExecutionPolicy(BaseExecutionPolicy):
    cpu_time_seconds: int | None = None
    memory_bytes: int | None = None
    create_process_group: bool = True


@dataclass
class CodexSandboxExecutionPolicy(BaseExecutionPolicy):
    binary: str = "codex"
    platform: typing.Literal["auto", "macos", "linux"] = "auto"
    config_overrides: Mapping[str, typing.Any] = field(default_factory=dict)


@dataclass
class DockerExecutionPolicy(BaseExecutionPolicy):
    binary: str = "docker"
    image: str = "python:3.12-alpine3.19"
    remove_container_on_exit: bool = True
    network_enabled: bool = False
    extra_run_args: Sequence[str] | None = None
    memory_bytes: int | None = None
    cpus: str | None = None
    read_only_rootfs: bool = False
    user: str | None = None
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain.agents.middleware._execution import (
    BaseExecutionPolicy,
    HostExecutionPolicy,
    CodexSandboxExecutionPolicy,
    DockerExecutionPolicy,
)
</syntaxhighlight>

== I/O Contract ==

=== BaseExecutionPolicy Parameters ===
{| class="wikitable"
|-
! Parameter
! Type
! Default
! Description
|-
| command_timeout
| float
| 30.0
| Maximum seconds for command execution
|-
| startup_timeout
| float
| 30.0
| Maximum seconds for shell startup
|-
| termination_timeout
| float
| 10.0
| Maximum seconds to wait for graceful termination
|-
| max_output_lines
| int
| 100
| Maximum output lines to capture
|-
| max_output_bytes
| int or None
| None
| Maximum output bytes to capture
|}

=== HostExecutionPolicy Parameters ===
{| class="wikitable"
|-
! Parameter
! Type
! Default
! Description
|-
| cpu_time_seconds
| int or None
| None
| CPU time limit via resource.RLIMIT_CPU
|-
| memory_bytes
| int or None
| None
| Memory limit via resource.RLIMIT_AS or RLIMIT_DATA
|-
| create_process_group
| bool
| True
| Create new process group for timeout handling
|}

=== CodexSandboxExecutionPolicy Parameters ===
{| class="wikitable"
|-
! Parameter
! Type
! Default
! Description
|-
| binary
| str
| "codex"
| Path to Codex CLI executable
|-
| platform
| "auto", "macos", "linux"
| "auto"
| Target platform for sandbox profile
|-
| config_overrides
| Mapping[str, Any]
| {}
| Codex CLI config overrides (-c flags)
|}

=== DockerExecutionPolicy Parameters ===
{| class="wikitable"
|-
! Parameter
! Type
! Default
! Description
|-
| binary
| str
| "docker"
| Path to Docker CLI executable
|-
| image
| str
| "python:3.12-alpine3.19"
| Docker image to run commands in
|-
| remove_container_on_exit
| bool
| True
| Pass --rm flag to docker run
|-
| network_enabled
| bool
| False
| Enable container networking (default: --network none)
|-
| memory_bytes
| int or None
| None
| Memory limit via --memory flag
|-
| cpus
| str or None
| None
| CPU quota via --cpus flag
|-
| read_only_rootfs
| bool
| False
| Mount rootfs as read-only
|-
| user
| str or None
| None
| Run as specific user via --user flag
|-
| extra_run_args
| Sequence[str] or None
| None
| Additional docker run arguments
|}

=== spawn() Method ===
{| class="wikitable"
|-
! Method
! Returns
! Description
|-
| spawn(workspace: Path, env: Mapping[str, str], command: Sequence[str])
| subprocess.Popen[str]
| Launch shell process with policy constraints
|}

== Usage Examples ==

=== Basic Host Execution ===
<syntaxhighlight lang="python">
from pathlib import Path
from langchain.agents.middleware._execution import HostExecutionPolicy

# Simple host execution with no resource limits
policy = HostExecutionPolicy()

# Spawn a bash shell
process = policy.spawn(
    workspace=Path("/tmp/workspace"),
    env={"PATH": "/usr/bin:/bin"},
    command=["/bin/bash", "-i"]
)

# Use the process for command execution
process.stdin.write("echo 'Hello World'\n")
process.stdin.flush()
output = process.stdout.readline()
print(output)
</syntaxhighlight>

=== Host Execution with Resource Limits ===
<syntaxhighlight lang="python">
from pathlib import Path
from langchain.agents.middleware._execution import HostExecutionPolicy

# Constrain CPU time and memory
policy = HostExecutionPolicy(
    command_timeout=60.0,
    cpu_time_seconds=300,  # 5 minutes of CPU time
    memory_bytes=512 * 1024 * 1024,  # 512 MB
    max_output_lines=500
)

process = policy.spawn(
    workspace=Path("/tmp/workspace"),
    env={"PYTHONPATH": "/usr/local/lib/python3.9"},
    command=["/bin/bash", "-i"]
)

# Commands will be killed if they exceed CPU or memory limits
</syntaxhighlight>

=== Codex Sandbox Execution ===
<syntaxhighlight lang="python">
from pathlib import Path
from langchain.agents.middleware._execution import CodexSandboxExecutionPolicy

# Use Codex CLI for syscall filtering
policy = CodexSandboxExecutionPolicy(
    platform="linux",  # or "macos" or "auto"
    config_overrides={
        "filesystem.writable": ["/tmp"],
        "network.allowed": False
    }
)

# Requires 'codex' binary on PATH
try:
    process = policy.spawn(
        workspace=Path("/tmp/workspace"),
        env={"HOME": "/tmp"},
        command=["/bin/bash", "-i"]
    )
    # Shell runs with Landlock/seccomp restrictions
except RuntimeError as e:
    print(f"Codex CLI not available: {e}")
</syntaxhighlight>

=== Docker Container Execution ===
<syntaxhighlight lang="python">
from pathlib import Path
from langchain.agents.middleware._execution import DockerExecutionPolicy

# Maximum isolation with Docker
policy = DockerExecutionPolicy(
    image="python:3.12-alpine3.19",
    network_enabled=False,
    memory_bytes=256 * 1024 * 1024,  # 256 MB
    cpus="0.5",  # Half a CPU
    read_only_rootfs=True,
    user="nobody",
    extra_run_args=["--cap-drop", "ALL"]
)

# Workspace is mounted only if not temporary
workspace = Path("/home/user/project")
process = policy.spawn(
    workspace=workspace,
    env={"LANG": "C.UTF-8"},
    command=["/bin/sh"]
)

# Commands run in isolated container
</syntaxhighlight>

=== Configuring Timeouts and Output Limits ===
<syntaxhighlight lang="python">
from pathlib import Path
from langchain.agents.middleware._execution import HostExecutionPolicy

# Fine-tune timeouts for specific use case
policy = HostExecutionPolicy(
    command_timeout=120.0,  # 2 minutes per command
    startup_timeout=10.0,   # Fast startup required
    termination_timeout=5.0,  # Quick cleanup
    max_output_lines=1000,  # Large output expected
    max_output_bytes=10 * 1024 * 1024  # 10 MB max
)

process = policy.spawn(
    workspace=Path("/tmp/workspace"),
    env={},
    command=["/bin/bash", "-i"]
)
</syntaxhighlight>

=== Using with Agent Middleware ===
<syntaxhighlight lang="python">
from pathlib import Path
from langchain.agents.middleware._execution import (
    HostExecutionPolicy,
    DockerExecutionPolicy
)

# Choose policy based on environment
def get_execution_policy(environment: str):
    if environment == "development":
        # Direct host access for development
        return HostExecutionPolicy(
            cpu_time_seconds=600,
            memory_bytes=1024 * 1024 * 1024  # 1 GB
        )
    elif environment == "production":
        # Strong isolation for production
        return DockerExecutionPolicy(
            image="python:3.12-slim",
            network_enabled=True,
            memory_bytes=512 * 1024 * 1024,
            cpus="1.0",
            user="appuser"
        )
    else:
        raise ValueError(f"Unknown environment: {environment}")


policy = get_execution_policy("production")
# Pass policy to middleware constructor
</syntaxhighlight>

=== Custom Docker Image with Tools ===
<syntaxhighlight lang="python">
from pathlib import Path
from langchain.agents.middleware._execution import DockerExecutionPolicy

# Use custom image with pre-installed tools
policy = DockerExecutionPolicy(
    image="myorg/agent-tools:v1.0",
    network_enabled=True,
    memory_bytes=512 * 1024 * 1024,
    extra_run_args=[
        "--env-file", "/path/to/.env",
        "-v", "/host/data:/data:ro",  # Mount read-only data
        "--cap-drop", "ALL",
        "--cap-add", "NET_BIND_SERVICE"
    ]
)

process = policy.spawn(
    workspace=Path("/tmp/workspace"),
    env={"API_KEY": "secret"},
    command=["/bin/bash", "-i"]
)
</syntaxhighlight>

=== Platform-Specific Codex Configuration ===
<syntaxhighlight lang="python">
import sys
from pathlib import Path
from langchain.agents.middleware._execution import CodexSandboxExecutionPolicy

# Auto-detect platform and configure accordingly
policy = CodexSandboxExecutionPolicy(
    platform="auto",  # Will detect Linux or macOS
    config_overrides={
        # Common config for both platforms
        "filesystem.readable": ["/usr", "/lib", "/bin"],
        "filesystem.writable": ["/tmp"],
        "network.allowed": False,
        # Platform-specific features handled automatically
    }
)

# Fallback if Codex not available
try:
    process = policy.spawn(
        workspace=Path("/tmp/workspace"),
        env={},
        command=["/bin/bash"]
    )
except RuntimeError:
    print("Codex CLI not found, falling back to host execution")
    from langchain.agents.middleware._execution import HostExecutionPolicy
    fallback_policy = HostExecutionPolicy()
    process = fallback_policy.spawn(
        workspace=Path("/tmp/workspace"),
        env={},
        command=["/bin/bash"]
    )
</syntaxhighlight>

== Related Pages ==
* [[langchain-ai_langchain_PersistentShellMiddleware]] - Uses execution policies for shell management
* [[principle::Defense in Depth]]
* [[principle::Principle of Least Privilege]]
* [[environment::Multi-Tenant Agent Systems]]
* [[environment::CI/CD Pipeline Security]]
