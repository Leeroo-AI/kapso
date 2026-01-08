# Implementation: Close_issue_action

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Awesome Windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|GitHub Script Action|https://github.com/actions/github-script]]
* [[source::Doc|GitHub REST API - Issues|https://docs.github.com/en/rest/issues/issues#update-an-issue]]
|-
! Domains
| [[domain::GitHub_Actions]], [[domain::GitHub_API]], [[domain::Issue_Management]]
|-
! Last Updated
| [[last_updated::2026-01-08 18:30 GMT]]
|}

== Overview ==

Wrapper document for closing GitHub Issues via the github-script action and REST API.

=== Description ===

This workflow step uses the `actions/github-script` action to execute JavaScript code that calls the GitHub REST API to update an issue's state to 'closed'. It handles both comment-triggered and manually-dispatched workflows by checking for the issue number in different context locations.

=== Usage ===

Use this pattern when you need to programmatically close GitHub Issues as part of an automation workflow. The github-script action provides a convenient JavaScript runtime with pre-authenticated GitHub API access.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/0PandaDEV/awesome-windows awesome-windows]
* '''File:''' .github/workflows/covert_to_pr.yml
* '''Lines:''' L136-147

=== Signature ===
<syntaxhighlight lang="yaml">
- name: Close Issue
  uses: actions/github-script@v7
  with:
    github-token: ${{ secrets.PAT }}
    script: |
      const issueNumber = context.payload.inputs
        ? context.payload.inputs.issue_number
        : context.issue.number;
      await github.rest.issues.update({
        owner: context.repo.owner,
        repo: context.repo.repo,
        issue_number: issueNumber,
        state: 'closed'
      });
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="yaml">
- uses: actions/github-script@v7
  with:
    github-token: ${{ secrets.PAT }}
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| github-token || str || Yes || PAT with issues:write permission
|-
| context.payload.inputs.issue_number || number || For dispatch || Issue number from manual trigger
|-
| context.issue.number || number || For comment || Issue number from comment event context
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Issue state || state || Issue state changed to 'closed'
|}

== Usage Examples ==

=== API Call Breakdown ===
<syntaxhighlight lang="javascript">
// github.rest.issues.update() wraps:
// PATCH /repos/{owner}/{repo}/issues/{issue_number}

await github.rest.issues.update({
  owner: context.repo.owner,     // e.g., "0PandaDEV"
  repo: context.repo.repo,       // e.g., "awesome-windows"
  issue_number: issueNumber,     // e.g., 42
  state: 'closed'                // Change state to closed
});

// Additional updateable fields (not used here):
// - title: string
// - body: string
// - labels: string[]
// - assignees: string[]
// - milestone: number
</syntaxhighlight>

=== Context Source Detection ===
<syntaxhighlight lang="javascript">
// Handle both trigger types:
const issueNumber = context.payload.inputs
  ? context.payload.inputs.issue_number   // workflow_dispatch
  : context.issue.number;                 // issue_comment

// context.payload.inputs exists only for workflow_dispatch
// context.issue.number exists only for issue-related events
</syntaxhighlight>

=== Complete Workflow Context ===
<syntaxhighlight lang="yaml">
# After PR is created, close the original submission issue
# This prevents duplicate tracking (issue + PR for same submission)

- name: Create Pull Request
  uses: peter-evans/create-pull-request@v6
  # ... creates PR with "Closes #N" in body

- name: Close Issue
  uses: actions/github-script@v7
  # ... immediately closes issue (PR body will close again on merge, but that's idempotent)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:0PandaDEV_Awesome_windows_Issue_State_Management]]

=== Requires Environment ===
* [[requires_env::Environment:0PandaDEV_Awesome_windows_GitHub_Actions_Ubuntu]]
