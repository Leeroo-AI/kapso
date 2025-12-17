{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|n8n Python Task Runner|https://docs.n8n.io]]
|-
! Domains
| [[domain::Security]], [[domain::Error_Handling]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Collecting and reporting all security violations found during code analysis in a comprehensive, actionable format.

=== Description ===

Violation reporting is the process of aggregating all security violations detected during static analysis and presenting them to the user in a single, detailed error report. Rather than failing on the first violation, the analyzer continues traversing the entire AST, collects all violations with their locations and descriptions, and then raises a single exception containing the complete violation list.

This principle solves the usability problem of incremental error reporting:
* Users need to see all problems at once to fix them efficiently
* Failing on first error requires multiple fix-submit cycles
* Line numbers help users locate violations in source code
* Comprehensive reports enable better security understanding

The violation report provides developers with complete context to understand and fix all security issues in one iteration.

=== Usage ===

Apply this principle when:
* Implementing code validators or linters
* Building security analysis tools
* Performing static analysis with multiple rules
* Providing developer-friendly error messages

== Theoretical Basis ==

Violation reporting implements error aggregation with deferred exception:

<pre>
Aggregation Pattern:
  class SecurityValidator:
    def __init__(self):
      self.violations = []  # Collect violations

    def record_violation(self, message, node):
      self.violations.append({
        'message': message,
        'line': node.lineno,
        'column': node.col_offset
      })

    def validate(self, code):
      tree = ast.parse(code)
      self.visit(tree)  # Traverse entire tree

      if self.violations:
        self.raise_security_error()

    def raise_security_error(self):
      report = format_violations(self.violations)
      raise SecurityError(report)

Violation Report Format:
  "Security validation failed with 3 violations:

  Line 5: Illegal import of 'os' module
  Line 12: Access to dangerous attribute '__globals__'
  Line 18: Use of blocked function 'eval'

  Allowed imports: json, math, datetime, re, ...
  "

Error Presentation:
  1. Summary: Number of violations found
  2. Details: Each violation with location
  3. Context: Security policy information
  4. Guidance: How to fix violations

Benefits:
  - Single analysis pass finds all violations
  - Developers fix all issues at once
  - Line numbers enable quick navigation
  - Complete context aids understanding
</pre>

The deferred exception approach balances thoroughness (finding all violations) with fail-fast behavior (preventing execution of invalid code).

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_TaskAnalyzer_raise_security_error]]

=== Related Principles ===
* [[related::Principle:n8n-io_n8n_Pattern_Detection]]
* [[related::Principle:n8n-io_n8n_Import_Validation]]
