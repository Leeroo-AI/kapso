# Implementation: huggingface_transformers_notification_service_doc_tests

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::CI_CD]], [[domain::Documentation]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Specialized notification service for doctest failures, posting formatted reports to documentation team channels.

=== Description ===

The `utils/notification_service_doc_tests.py` module (384 lines) handles doctest-specific notifications. It:
- Parses doctest failure output
- Formats failures with code context
- Links to specific documentation pages
- Posts to documentation team Slack channel

=== Usage ===

Called after doctest CI jobs to alert documentation maintainers about failing examples.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' [https://github.com/huggingface/transformers/blob/main/utils/notification_service_doc_tests.py utils/notification_service_doc_tests.py]
* '''Lines:''' 1-384

=== Signature ===
<syntaxhighlight lang="python">
def parse_doctest_failures(output: str) -> list[dict]:
    """Extract failures from doctest output."""

def format_doctest_message(failures: list[dict]) -> dict:
    """Create Slack message for doctest failures."""

def main():
    """Process doctest results and notify."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="bash">
python utils/notification_service_doc_tests.py --output doctest_output.txt
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| Doctest output || File || Yes || Raw doctest results
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Slack notification || HTTP || Posted to docs channel
|}

== Usage Examples ==

=== CI Integration ===
<syntaxhighlight lang="yaml">
- name: Notify Doctest Failures
  if: failure()
  run: python utils/notification_service_doc_tests.py
</syntaxhighlight>

== Related Pages ==
