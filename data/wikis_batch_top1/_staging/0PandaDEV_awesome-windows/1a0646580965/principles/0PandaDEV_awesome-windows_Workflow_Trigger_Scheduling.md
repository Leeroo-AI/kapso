# Principle: Workflow Trigger Scheduling

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|GitHub Actions Scheduled Events|https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#schedule]]
|-
! Domains
| [[domain::CI_CD]], [[domain::Automation]], [[domain::Scheduling]]
|-
! Last Updated
| [[last_updated::2026-01-08 13:00 GMT]]
|}

== Overview ==

Principle for configuring time-based and manual triggers for automated GitHub Actions workflows.

=== Description ===

Workflow Trigger Scheduling enables automation tasks to run on a predictable schedule without manual intervention. In the context of awesome-windows, this principle governs when the contributor update workflow executes.

GitHub Actions supports multiple trigger types:
- '''schedule''' - Cron-based time triggers
- '''workflow_dispatch''' - Manual trigger with optional inputs
- '''push/pull_request''' - Event-based triggers

The contributor update workflow uses both scheduled and manual triggers to balance automation with on-demand control.

=== Usage ===

Apply this principle when designing automation that should run:
- Periodically (daily, weekly, etc.)
- On-demand via manual trigger
- In response to repository events

For the contributor update workflow:
- Schedule: Daily at midnight UTC
- Manual: Via Actions UI or API

== Theoretical Basis ==

'''Cron Expression Format:'''

<syntaxhighlight lang="text">
┌───────────── minute (0 - 59)
│ ┌───────────── hour (0 - 23)
│ │ ┌───────────── day of month (1 - 31)
│ │ │ ┌───────────── month (1 - 12)
│ │ │ │ ┌───────────── day of week (0 - 6) (Sunday to Saturday)
│ │ │ │ │
│ │ │ │ │
* * * * *
</syntaxhighlight>

'''Common Patterns:'''

{| class="wikitable"
|-
! Cron !! Meaning
|-
| `0 0 * * *` || Daily at midnight UTC
|-
| `0 */6 * * *` || Every 6 hours
|-
| `0 0 * * 0` || Weekly on Sunday
|-
| `0 0 1 * *` || Monthly on the 1st
|}

'''Trigger Types Comparison:'''

{| class="wikitable"
|-
! Trigger !! When !! Use Case
|-
| schedule || Time-based (cron) || Regular maintenance tasks
|-
| workflow_dispatch || Manual || On-demand execution
|-
| push || Code changes || Build/test on commits
|-
| pull_request || PR events || Validation before merge
|}

'''GitHub Actions Schedule Limitations:'''
- Minimum interval: ~5 minutes (shorter schedules may be throttled)
- Time zone: Always UTC
- Reliability: May be delayed during high GitHub load
- Skipping: Runs skipped if repository inactive for 60+ days

== Practical Guide ==

=== Choosing Schedule Frequency ===

Consider:
- '''Data volatility:''' How often does source data change?
- '''Cost:''' More runs = more Action minutes
- '''Freshness:''' How current must the data be?

For contributor updates:
- Contributors change infrequently
- Daily check is sufficient
- Midnight UTC minimizes conflicts

=== Manual Override Option ===

Always include `workflow_dispatch` for:
- Testing the workflow
- Forcing immediate updates
- Debugging issues

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:0PandaDEV_awesome-windows_GitHub_Actions_Cron_Schedule]]

=== Part of Workflow ===
* [[workflow::Workflow:0PandaDEV_awesome-windows_Automated_Contributor_Update]]
