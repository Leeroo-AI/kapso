# Principle: Violation Aggregation

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

Principle for collecting multiple security violations during analysis and reporting them in a structured, actionable error format.

=== Description ===

Violation Aggregation ensures users receive comprehensive security feedback:

1. **Collection**: All violations gathered during AST traversal (not fail-fast)
2. **Formatting**: Each violation includes line number and descriptive message
3. **Aggregation**: Multiple violations combined with newline separators
4. **Error Packaging**: Final error contains message and detailed description

Benefits:
- **Complete Feedback**: Users see all issues at once, not one at a time
- **Actionable Errors**: Line numbers enable quick navigation to problems
- **Clear Communication**: Descriptive messages explain what and why
- **Caching Support**: Violations list can be cached and reused

=== Usage ===

Apply this principle when:
- Building validators that may find multiple issues
- Designing user-facing error reporting
- Creating linters or code analysis tools
- Implementing security systems with detailed feedback

== Theoretical Basis ==

Violation aggregation follows a **Collect-Then-Report** pattern:

<syntaxhighlight lang="python">
# Pseudo-code for violation aggregation

class SecurityValidator:
    def __init__(self):
        self.violations: list[str] = []  # Collect during traversal

    def _add_violation(self, lineno: int, message: str):
        self.violations.append(f"Line {lineno}: {message}")

# After traversal completes
if security_validator.violations:
    description = "\n".join(security_validator.violations)
    raise SecurityViolationError(
        message="Security violations detected",
        description=description
    )

# Example description output:
# "Line 2: Import 'os' is not allowed. Allowed stdlib modules: json, datetime\n
#  Line 5: Access to dangerous attribute '__class__' is not allowed\n
#  Line 8: Dynamic import is not allowed"
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_SecurityViolationError]]
