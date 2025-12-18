# Implementation: TaskAnalyzer.validate

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Security]], [[domain::Static_Analysis]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Concrete method for validating Python code against security policies using AST-based static analysis with result caching.

=== Description ===

`TaskAnalyzer.validate()` is the primary security gate before code execution. It:

1. **Fast-path for permissive config**: Skips validation if `*` wildcard is in both stdlib and external allowlists
2. **Cache lookup**: Checks if this code+allowlists combination was previously validated
3. **AST parsing**: Parses code string into Abstract Syntax Tree
4. **SecurityValidator traversal**: Walks AST to detect violations
5. **Cache storage**: Stores validation result for future lookups
6. **Error raising**: Throws `SecurityViolationError` with detailed violation descriptions

The cache uses SHA-256 hashing of code and a sorted tuple of allowlists as the key, with FIFO eviction at 500 entries.

=== Usage ===

Call this method before executing any user-provided code. It raises `SecurityViolationError` if violations are detected, or returns silently if code is safe.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/task-runner-python/src/task_analyzer.py
* '''Lines:''' L172-196

=== Signature ===
<syntaxhighlight lang="python">
def validate(self, code: str) -> None:
    """
    Validate Python code against security policies.

    Args:
        code: Python source code to validate.

    Raises:
        SecurityViolationError: If code violates security policies.
            Contains 'description' with newline-separated violation details.
        SyntaxError: If code cannot be parsed.

    Returns:
        None if code passes validation.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from src.task_analyzer import TaskAnalyzer
from src.config.security_config import SecurityConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| code || str || Yes || Python source code to validate
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| (returns) || None || Returns silently if code is valid
|-
| (raises) || SecurityViolationError || If violations detected, contains description field with details
|-
| (raises) || SyntaxError || If code cannot be parsed
|}

== Usage Examples ==

=== Basic Validation ===
<syntaxhighlight lang="python">
from src.task_analyzer import TaskAnalyzer
from src.config.security_config import SecurityConfig
from src.errors import SecurityViolationError

# Create analyzer with restricted allowlist
security_config = SecurityConfig(
    stdlib_allow={"json", "datetime"},
    external_allow=set(),
    builtins_deny={"eval", "exec", "compile"},
    runner_env_deny=True,
)
analyzer = TaskAnalyzer(security_config)

# Valid code
safe_code = """
import json
data = json.loads('{"key": "value"}')
return data
"""
analyzer.validate(safe_code)  # Returns None

# Invalid code - blocked import
unsafe_code = """
import os
os.system("rm -rf /")
"""
try:
    analyzer.validate(unsafe_code)
except SecurityViolationError as e:
    print(f"Blocked: {e.description}")
    # Output: Line 2: Import 'os' is not allowed...
</syntaxhighlight>

=== Detecting Dangerous Patterns ===
<syntaxhighlight lang="python">
# Dangerous attribute access
dunder_code = """
x = ().__class__.__bases__[0].__subclasses__()
"""
try:
    analyzer.validate(dunder_code)
except SecurityViolationError as e:
    print(e.description)
    # Output: Line 2: Access to dangerous attribute '__class__' is not allowed
    #         Line 2: Access to dangerous attribute '__bases__' is not allowed
    #         Line 2: Access to dangerous attribute '__subclasses__' is not allowed
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Security_Validation]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Task_Runner_Env]]
