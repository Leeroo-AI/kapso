# Implementation: SecurityViolationError

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Security]], [[domain::Error_Handling]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Concrete exception class for reporting security violations with a message and detailed description.

=== Description ===

`SecurityViolationError` is a custom exception that carries:

1. **message**: Short summary message (default: "Security violations detected")
2. **description**: Detailed multi-line description of all violations

The error is raised by `TaskAnalyzer.validate()` when code violates security policies and is caught by `TaskRunner._execute_task()` to send error details back to the broker.

=== Usage ===

This exception is raised internally by the security validation system. Catch it to extract violation details for user feedback.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/task-runner-python/src/errors/security_violation_error.py
* '''Lines:''' L1-9

=== Signature ===
<syntaxhighlight lang="python">
class SecurityViolationError(Exception):
    """Raised when code violates security policies, typically through
    the use of disallowed modules or builtins."""

    def __init__(
        self,
        message: str = "Security violations detected",
        description: str = ""
    ):
        """
        Args:
            message: Short error summary.
            description: Detailed violation list (typically newline-separated).
        """
        super().__init__(message)
        self.message = message
        self.description = description
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from src.errors import SecurityViolationError
# or
from src.errors.security_violation_error import SecurityViolationError
</syntaxhighlight>

== I/O Contract ==

=== Constructor Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| message || str || No || Short error message (default: "Security violations detected")
|-
| description || str || No || Detailed violation list (default: "")
|}

=== Attributes ===
{| class="wikitable"
|-
! Attribute !! Type !! Description
|-
| message || str || Short error summary
|-
| description || str || Newline-separated violation details with line numbers
|}

== Usage Examples ==

=== Raising the Error ===
<syntaxhighlight lang="python">
from src.errors import SecurityViolationError

# In TaskAnalyzer._raise_security_error()
def _raise_security_error(self, violations: list[str]) -> None:
    raise SecurityViolationError(
        message="Security violations detected",
        description="\n".join(violations)
    )

# Example violations list:
# ["Line 2: Import 'os' is not allowed. Allowed stdlib modules: json",
#  "Line 5: Access to dangerous attribute '__class__' is not allowed"]

# Results in error with:
# message = "Security violations detected"
# description = "Line 2: Import 'os' is not allowed. Allowed stdlib modules: json\n
#               Line 5: Access to dangerous attribute '__class__' is not allowed"
</syntaxhighlight>

=== Catching in TaskRunner ===
<syntaxhighlight lang="python">
# In TaskRunner._execute_task()
async def _execute_task(self, task_id, task_settings):
    try:
        self.analyzer.validate(task_settings.code)
        # ... execution ...
    except SecurityViolationError as e:
        # Extract both message and description for broker
        error = {
            "message": e.message,        # "Security violations detected"
            "description": e.description  # Detailed line-by-line violations
        }
        response = RunnerTaskError(task_id=task_id, error=error)
        await self._send_message(response)
</syntaxhighlight>

=== Also Raised at Runtime ===
<syntaxhighlight lang="python">
# In TaskExecutor._create_safe_import()
def safe_import(name, *args, **kwargs):
    is_allowed, error_msg = validate_module_import(name, security_config)

    if not is_allowed:
        # Runtime import blocked
        raise SecurityViolationError(
            message="Security violation detected",
            description=error_msg,
        )

    return original_import(name, *args, **kwargs)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:n8n-io_n8n_Violation_Aggregation]]

=== Requires Environment ===
* [[requires_env::Environment:n8n-io_n8n_Python_Task_Runner_Env]]
