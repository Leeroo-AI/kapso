# Implementation: Convert_command_check

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Awesome Windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|GitHub Actions Events|https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#issue_comment]]
* [[source::Doc|GitHub Actions Expressions|https://docs.github.com/en/actions/learn-github-actions/expressions]]
|-
! Domains
| [[domain::GitHub_Actions]], [[domain::Event_Handling]], [[domain::Authorization]]
|-
! Last Updated
| [[last_updated::2026-01-08 18:30 GMT]]
|}

== Overview ==

Pattern document defining the comment-based command trigger for issue-to-PR conversion.

=== Description ===

This GitHub Actions conditional pattern triggers the issue-to-PR conversion workflow when specific conditions are met. It checks for the exact comment `/convert`, verifies the comment author is the repository owner (`0pandadev`), and confirms the issue has the "Add" label. It also supports manual triggering via `workflow_dispatch`.

=== Usage ===

Use this pattern when implementing chat-ops style commands in GitHub repositories. The multi-condition `if` statement provides authorization control (owner-only) and context validation (correct label) for automated actions triggered by issue comments.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/0PandaDEV/awesome-windows awesome-windows]
* '''File:''' .github/workflows/covert_to_pr.yml
* '''Lines:''' L15-20

=== Signature ===
<syntaxhighlight lang="yaml">
jobs:
  convert_issue_to_pr:
    if: |
      (github.event_name == 'issue_comment' &&
      github.event.comment.body == '/convert' &&
      github.event.comment.user.login == '0pandadev' &&
      contains(github.event.issue.labels.*.name, 'Add')) ||
      github.event_name == 'workflow_dispatch'
    runs-on: ubuntu-latest
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="yaml">
on:
  issue_comment:
    types: [created]
  workflow_dispatch:
    inputs:
      issue_number:
        description: 'Issue number to convert to PR'
        required: true
        type: number
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| github.event.comment.body || str || Yes || Must exactly equal `/convert`
|-
| github.event.comment.user.login || str || Yes || Must equal `0pandadev` (owner)
|-
| github.event.issue.labels.*.name || array || Yes || Must contain "Add" label
|-
| github.event.inputs.issue_number || number || For dispatch || Manual trigger issue number
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Job execution || boolean || Job runs if conditions met, skipped otherwise
|}

== Usage Examples ==

=== Condition Breakdown ===
<syntaxhighlight lang="yaml">
# This job runs when EITHER:
# 1. All comment conditions are true:
#    - Event is issue_comment
#    - Comment body is exactly "/convert"
#    - Commenter is "0pandadev"
#    - Issue has "Add" label
# OR
# 2. Manual workflow_dispatch trigger

if: |
  (github.event_name == 'issue_comment' &&
  github.event.comment.body == '/convert' &&
  github.event.comment.user.login == '0pandadev' &&
  contains(github.event.issue.labels.*.name, 'Add')) ||
  github.event_name == 'workflow_dispatch'
</syntaxhighlight>

=== Usage Scenario ===
<syntaxhighlight lang="text">
1. User submits app via Issue Form -> Issue created with "Add" label
2. Maintainer reviews submission
3. Maintainer comments "/convert" on the issue
4. Workflow triggers (if maintainer is 0pandadev)
5. Issue is converted to Pull Request
</syntaxhighlight>

=== Manual Trigger ===
<syntaxhighlight lang="yaml">
# For manual dispatch, provide issue number in UI
workflow_dispatch:
  inputs:
    issue_number:
      description: 'Issue number to convert to PR'
      required: true
      type: number
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:0PandaDEV_Awesome_windows_Comment_Command_Trigger]]

=== Requires Environment ===
* [[requires_env::Environment:0PandaDEV_Awesome_windows_GitHub_Actions_Ubuntu]]
