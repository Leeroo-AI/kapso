# Implementation: SecurityConfig

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Security]], [[domain::Configuration]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Concrete dataclass defining security policy settings for sandboxed Python code execution.

=== Description ===

`SecurityConfig` is a simple dataclass that holds security policy settings:

- **`stdlib_allow`**: Set of standard library module names permitted for import
- **`external_allow`**: Set of external package names permitted for import
- **`builtins_deny`**: Set of builtin function names to remove from execution context
- **`runner_env_deny`**: Boolean flag to clear environment variables before execution

The dataclass is passed through the execution pipeline to `TaskAnalyzer` for static validation and `TaskExecutor` for runtime enforcement.

=== Usage ===

Create an instance directly or via `TaskRunnerConfig.from_env()` which reads security settings from environment variables.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/task-runner-python/src/config/security_config.py
* '''Lines:''' L4-9

=== Signature ===
<syntaxhighlight lang="python">
from dataclasses import dataclass

@dataclass
class SecurityConfig:
    stdlib_allow: set[str]
    external_allow: set[str]
    builtins_deny: set[str]
    runner_env_deny: bool
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from src.config.security_config import SecurityConfig
</syntaxhighlight>

== I/O Contract ==

=== Fields ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| stdlib_allow || set[str] || Allowed standard library modules. Use {"*"} for all.
|-
| external_allow || set[str] || Allowed external packages. Use {"*"} for all.
|-
| builtins_deny || set[str] || Builtin functions to remove from __builtins__.
|-
| runner_env_deny || bool || If True, clear os.environ before code execution.
|}

=== Environment Variables (via TaskRunnerConfig) ===
{| class="wikitable"
|-
! Env Var !! Maps To !! Format
|-
| N8N_RUNNERS_STDLIB_ALLOW || stdlib_allow || Comma-separated module names or "*"
|-
| N8N_RUNNERS_EXTERNAL_ALLOW || external_allow || Comma-separated package names or "*"
|-
| N8N_RUNNERS_BUILTINS_DENY || builtins_deny || Comma-separated builtin names
|-
| N8N_BLOCK_RUNNER_ENV_ACCESS || runner_env_deny || "true" or "false"
|}

== Usage Examples ==

=== Restrictive Configuration ===
<syntaxhighlight lang="python">
from src.config.security_config import SecurityConfig

# Minimal permissions - only JSON and datetime
restrictive_config = SecurityConfig(
    stdlib_allow={"json", "datetime", "re"},
    external_allow=set(),  # No external packages
    builtins_deny={"eval", "exec", "compile", "open", "input"},
    runner_env_deny=True,
)
</syntaxhighlight>

=== Permissive Configuration ===
<syntaxhighlight lang="python">
# All modules allowed (for trusted environments)
permissive_config = SecurityConfig(
    stdlib_allow={"*"},  # All stdlib
    external_allow={"*"},  # All external
    builtins_deny=set(),  # No restrictions
    runner_env_deny=False,  # Keep environment
)
</syntaxhighlight>

=== Data Science Configuration ===
<syntaxhighlight lang="python">
# Common data science setup
data_science_config = SecurityConfig(
    stdlib_allow={"json", "datetime", "re", "math", "statistics", "collections"},
    external_allow={"pandas", "numpy", "scipy"},
    builtins_deny={"eval", "exec", "compile", "__import__"},
    runner_env_deny=True,
)
</syntaxhighlight>

=== From Environment Variables ===
<syntaxhighlight lang="python">
# In production, config comes from environment
from src.config.task_runner_config import TaskRunnerConfig

# Reads N8N_RUNNERS_* environment variables
config = TaskRunnerConfig.from_env()

# SecurityConfig is created in TaskRunner.__init__
security_config = SecurityConfig(
    stdlib_allow=config.stdlib_allow,
    external_allow=config.external_allow,
    builtins_deny=config.builtins_deny,
    runner_env_deny=config.env_deny,
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Security_Configuration]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Task_Runner_Env]]
