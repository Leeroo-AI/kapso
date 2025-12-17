# Implementation: huggingface_transformers_notification_service

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::CI_CD]], [[domain::Notifications]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

CI notification service that posts test failure reports to Slack channels with detailed failure analysis and attribution.

=== Description ===

The `utils/notification_service.py` module (1622 lines) handles CI result reporting. It:
- Aggregates test failures from CI runs
- Formats failure reports with stack traces
- Attributes failures to responsible commits/authors
- Posts formatted messages to Slack channels
- Supports different notification levels (critical, warning, info)

=== Usage ===

Called automatically by CI after test runs complete. Configured via environment variables for Slack webhook URLs.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' [https://github.com/huggingface/transformers/blob/main/utils/notification_service.py utils/notification_service.py]
* '''Lines:''' 1-1622

=== Signature ===
<syntaxhighlight lang="python">
def parse_test_results(results_dir: str) -> dict:
    """Parse pytest results from JUnit XML."""

def format_slack_message(failures: list[dict]) -> dict:
    """Create Slack block message from failures."""

def post_to_slack(message: dict, webhook_url: str) -> None:
    """Send message to Slack channel."""

def main():
    """
    Process test results and send notifications.

    Env vars:
        SLACK_WEBHOOK_URL: Slack incoming webhook
        CI_JOB_NAME: Current CI job name
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="bash">
python utils/notification_service.py --results_dir ./test-results
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| Test results || XML/JSON || Yes || JUnit test result files
|-
| SLACK_WEBHOOK_URL || env || Yes || Slack webhook
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Slack message || HTTP || Posted to Slack channel
|}

== Usage Examples ==

=== CI Integration ===
<syntaxhighlight lang="yaml">
# GitHub Actions workflow
- name: Notify Slack
  if: failure()
  run: python utils/notification_service.py
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
</syntaxhighlight>

== Related Pages ==
