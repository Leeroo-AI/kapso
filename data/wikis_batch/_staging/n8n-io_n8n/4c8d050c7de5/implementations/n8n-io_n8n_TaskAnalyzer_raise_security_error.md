{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Security]], [[domain::Error_Handling]], [[domain::Reporting]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Concrete tool for reporting security violations by raising a structured exception containing all detected violations from AST analysis.

=== Description ===

The `_raise_security_error()` method is a private helper in the `TaskAnalyzer` class that aggregates all security violations detected during static code analysis and raises a `SecurityViolationError` exception. This provides a clear, structured way to communicate security failures to the caller.

=== Usage ===

This implementation is invoked at the end of the validation process if any violations were detected by the `SecurityValidator` during AST traversal. It converts the internal violations list into a user-facing exception that includes detailed descriptions of all security issues found, preventing task execution and informing the user of the problems.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/n8n-io/n8n n8n]
* '''File:''' packages/@n8n/task-runner-python/src/task_analyzer.py
* '''Lines:''' L198-201

=== Signature ===
<syntaxhighlight lang="python">
def _raise_security_error(self, violations: list[str]) -> None:
    """Raise SecurityViolationError with formatted violation messages."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from src.task_analyzer import TaskAnalyzer
from src.exceptions import SecurityViolationError
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| violations || list[str] || Yes || List of violation messages from SecurityValidator
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| (none) || Never returns || Method always raises exception
|}

=== Exceptions ===
{| class="wikitable"
|-
! Exception !! Condition
|-
| SecurityViolationError || Always raised when method is called
|}

== Implementation Details ==

=== Complete Implementation ===
<syntaxhighlight lang="python">
def _raise_security_error(self, violations: list[str]) -> None:
    raise SecurityViolationError(
        message="Security violations detected",
        description="\n".join(violations)
    )
</syntaxhighlight>

=== SecurityViolationError Structure ===
The exception uses a structured format:
* '''message''': High-level summary ("Security violations detected")
* '''description''': Detailed, newline-separated list of all violations

This structure allows:
* Brief error display in logs (using `message`)
* Detailed user feedback (using `description`)
* Programmatic parsing of individual violations

=== Violation Message Format ===
Each violation in the list follows the pattern:
<syntaxhighlight lang="text">
Line {lineno}: {error_message}
</syntaxhighlight>

Example violations list:
<syntaxhighlight lang="python">
violations = [
    "Line 5: Module 'os' is not in the allowlist",
    "Line 12: Access to dangerous attribute '__globals__' is not allowed",
    "Line 18: Usage of dangerous name 'eval' is not allowed"
]
</syntaxhighlight>

After joining with newlines:
<syntaxhighlight lang="text">
Line 5: Module 'os' is not in the allowlist
Line 12: Access to dangerous attribute '__globals__' is not allowed
Line 18: Usage of dangerous name 'eval' is not allowed
</syntaxhighlight>

=== Caller Pattern ===
The typical calling pattern in `TaskAnalyzer`:
<syntaxhighlight lang="python">
def validate_code(self, code: str) -> None:
    # Parse code
    tree = ast.parse(code)

    # Run security validation
    validator = SecurityValidator(self.security_config)
    validator.visit(tree)

    # Check for violations
    if validator.violations:
        self._raise_security_error(validator.violations)

    # If we reach here, code is safe
</syntaxhighlight>

=== Error Propagation ===
The exception propagates up the call stack:
<syntaxhighlight lang="text">
TaskAnalyzer.validate_code()
  └─> _raise_security_error()
        └─> raises SecurityViolationError
              └─> caught by TaskExecutor
                    └─> sent back to n8n broker as task error
</syntaxhighlight>

== Usage Examples ==

=== Basic Usage ===
<syntaxhighlight lang="python">
from src.task_analyzer import TaskAnalyzer
from src.exceptions import SecurityViolationError

analyzer = TaskAnalyzer()

code = """
import os
import sys
result = eval("1 + 1")
"""

try:
    analyzer.validate_code(code)
    print("Code passed validation")
except SecurityViolationError as e:
    print(f"Error: {e.message}")
    print(f"Details:\n{e.description}")

# Output:
# Error: Security violations detected
# Details:
# Line 1: Module 'os' is not in the allowlist
# Line 2: Module 'sys' is not in the allowlist
# Line 3: Usage of dangerous name 'eval' is not allowed
