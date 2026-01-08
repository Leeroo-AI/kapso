# Implementation: Create_pull_request_action

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Awesome Windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|peter-evans/create-pull-request|https://github.com/peter-evans/create-pull-request]]
|-
! Domains
| [[domain::GitHub_Actions]], [[domain::Pull_Requests]], [[domain::Automation]]
|-
! Last Updated
| [[last_updated::2026-01-08 18:30 GMT]]
|}

== Overview ==

Wrapper document for the peter-evans/create-pull-request GitHub Action used to create PRs from workflow changes.

=== Description ===

The `peter-evans/create-pull-request` action automatically creates a pull request when files are modified during a workflow run. It handles branch creation, commit generation, and PR opening in a single step. In this repository, it's used to create PRs from issue-to-PR conversions, with dynamic title and body content populated from parsed issue data.

=== Usage ===

Use this action when your GitHub Actions workflow modifies files and you want to propose those changes via a pull request rather than direct commits. It's particularly useful for automation workflows that require human review before merging.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/0PandaDEV/awesome-windows awesome-windows]
* '''File:''' .github/workflows/covert_to_pr.yml
* '''Lines:''' L120-134

=== Signature ===
<syntaxhighlight lang="yaml">
- name: Create Pull Request
  uses: peter-evans/create-pull-request@v6
  with:
    token: ${{ secrets.PAT }}
    commit-message: "Add ${{ env.APP_NAME }} to ${{ env.CATEGORY }} category"
    title: "Add ${{ env.APP_NAME }} to ${{ env.CATEGORY }} category"
    body: |
      This PR adds ${{ env.APP_NAME }} to the ${{ env.CATEGORY }} category in the README.md file.

      Application URL: ${{ env.APP_URL }}
      Repository URL: ${{ env.REPO_URL }}

      Closes #${{ github.event.issue.number || github.event.inputs.issue_number }}
    branch: add-${{ github.event.issue.number || github.event.inputs.issue_number }}
    base: main
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="yaml">
- uses: peter-evans/create-pull-request@v6
  with:
    token: ${{ secrets.PAT }}  # PAT required for workflow triggers
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| token || str || Yes || GitHub PAT with repo and workflow permissions
|-
| commit-message || str || Yes || Message for the auto-created commit
|-
| title || str || Yes || Pull request title
|-
| body || str || No || Pull request description body
|-
| branch || str || No || Name for the new branch (default: create-pull-request/patch)
|-
| base || str || No || Target branch for PR (default: repo default branch)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| pull-request-number || number || The PR number if created
|-
| pull-request-url || str || Full URL to the created PR
|-
| pull-request-operation || str || 'created', 'updated', or 'closed'
|}

== Usage Examples ==

=== Dynamic PR Content ===
<syntaxhighlight lang="yaml">
# Environment variables from earlier steps:
# APP_NAME=Visual Studio Code
# CATEGORY=IDEs
# APP_URL=https://code.visualstudio.com
# REPO_URL=https://github.com/microsoft/vscode

- name: Create Pull Request
  uses: peter-evans/create-pull-request@v6
  with:
    token: ${{ secrets.PAT }}
    commit-message: "Add Visual Studio Code to IDEs category"
    title: "Add Visual Studio Code to IDEs category"
    body: |
      This PR adds Visual Studio Code to the IDEs category in the README.md file.

      Application URL: https://code.visualstudio.com
      Repository URL: https://github.com/microsoft/vscode

      Closes #42
    branch: add-42
    base: main
</syntaxhighlight>

=== Issue Auto-Close ===
<syntaxhighlight lang="markdown">
# The "Closes #N" syntax in the PR body automatically closes
# the linked issue when the PR is merged.

Closes #${{ github.event.issue.number || github.event.inputs.issue_number }}

# Result: When PR merges, issue #42 is automatically closed
</syntaxhighlight>

=== Branch Naming Convention ===
<syntaxhighlight lang="yaml">
# Branch name includes issue number for traceability
branch: add-${{ github.event.issue.number }}

# Results in branches like:
# - add-42
# - add-108
# - add-255
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:0PandaDEV_Awesome_windows_PR_Creation]]

=== Requires Environment ===
* [[requires_env::Environment:0PandaDEV_Awesome_windows_GitHub_Actions_Ubuntu]]
