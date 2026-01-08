# Implementation: GitHub Actions Cron Schedule

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|GitHub Actions Schedule|https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#schedule]]
|-
! Domains
| [[domain::CI_CD]], [[domain::GitHub_Actions]], [[domain::Scheduling]]
|-
! Last Updated
| [[last_updated::2026-01-08 13:00 GMT]]
|}

== Overview ==

External tool documentation for GitHub Actions workflow trigger configuration using cron scheduling and manual dispatch.

=== Description ===

This is an '''External Tool Doc''' describing how the contributor update workflow configures its triggers using GitHub Actions' `on.schedule` and `on.workflow_dispatch` events.

=== Usage ===

The workflow runs automatically at midnight UTC daily. It can also be triggered manually via the Actions tab in GitHub or via the GitHub API.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/0PandaDEV/awesome-windows awesome-windows]
* '''File:''' .github/workflows/update_contributors.yml:L1-6

=== External Reference ===
* [https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#schedule GitHub Actions Schedule Documentation]

=== Trigger Configuration ===

<syntaxhighlight lang="yaml">
name: Update Contributors

on:
  schedule:
    - cron: "0 0 * * *"   # Daily at midnight UTC
  workflow_dispatch:       # Manual trigger (no inputs required)
</syntaxhighlight>

=== Cron Expression Breakdown ===

<syntaxhighlight lang="text">
"0 0 * * *"
 │ │ │ │ │
 │ │ │ │ └── Day of week: * (every day)
 │ │ │ └──── Month: * (every month)
 │ │ └────── Day of month: * (every day)
 │ └──────── Hour: 0 (midnight)
 └────────── Minute: 0 (on the hour)

= "Run at 00:00 UTC every day"
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| Cron schedule || String || Yes || Cron expression defining when to run
|-
| workflow_dispatch event || Webhook || No || Manual trigger from GitHub UI/API
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Workflow run initiated || Event || Starts the update-contributors job
|-
| github.event_name || String || "schedule" or "workflow_dispatch"
|}

== Usage Examples ==

=== Example 1: Automatic Daily Execution ===
<syntaxhighlight lang="text">
Schedule: 0 0 * * * (midnight UTC)

Timeline:
- 00:00 UTC: Workflow triggered by schedule
- 00:00 UTC: Job "update-contributors" starts
- 00:01 UTC: Script completes, README updated (if changes)
- 00:01 UTC: Commit pushed (if changes)

Actions Tab shows:
Run #42 · schedule · main · 2 minutes ago
</syntaxhighlight>

=== Example 2: Manual Trigger ===
<syntaxhighlight lang="text">
1. Navigate to Actions tab
2. Select "Update Contributors" workflow
3. Click "Run workflow" dropdown
4. Select branch: main
5. Click "Run workflow" button

Actions Tab shows:
Run #43 · workflow_dispatch · main · just now
</syntaxhighlight>

=== Example 3: API-Based Trigger ===
<syntaxhighlight lang="bash">
# Trigger workflow via GitHub API
curl -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/0PandaDEV/awesome-windows/actions/workflows/update_contributors.yml/dispatches \
  -d '{"ref":"main"}'
</syntaxhighlight>

=== Example 4: Viewing Schedule History ===
<syntaxhighlight lang="text">
Actions Tab → Update Contributors

Recent Runs:
✅ Run #42 · schedule · main · 2 hours ago      (midnight UTC)
✅ Run #41 · schedule · main · 26 hours ago     (midnight UTC yesterday)
✅ Run #40 · workflow_dispatch · main · 2 days  (manual trigger)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:0PandaDEV_awesome-windows_Workflow_Trigger_Scheduling]]

=== Requires Environment ===
* [[requires_env::Environment:0PandaDEV_awesome-windows_GitHub_Actions_Environment]]
