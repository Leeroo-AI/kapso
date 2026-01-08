# Principle: PR_Creation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|GitHub REST API - Pulls|https://docs.github.com/en/rest/pulls/pulls]]
* [[source::Doc|peter-evans/create-pull-request|https://github.com/peter-evans/create-pull-request]]
|-
! Domains
| [[domain::GitHub]], [[domain::Pull_Requests]], [[domain::Automation]]
|-
! Last Updated
| [[last_updated::2026-01-08 18:30 GMT]]
|}

== Overview ==

Technique for programmatically creating GitHub Pull Requests from automated workflows.

=== Description ===

PR Creation automates the process of proposing code changes via GitHub Pull Requests. This involves creating a new branch, committing changes, and opening a PR against a base branch. Automated PR creation enables workflows to propose changes for human review rather than directly committing to protected branches.

Key aspects include branch naming conventions, PR title and body formatting, linking to related issues, and handling authentication for cross-workflow triggers.

=== Usage ===

Use this principle when:
- Automation should propose changes rather than commit directly
- Changes require human review before merging
- Building bots that maintain documentation or dependencies
- Implementing issue-to-PR conversion workflows

== Theoretical Basis ==

=== PR Components ===
<syntaxhighlight lang="text">
Required:
- base: Target branch (e.g., main)
- head: Source branch with changes
- title: PR title string

Optional but recommended:
- body: Description with context
- labels: Categorization
- reviewers: Auto-assign reviewers
- milestone: Project tracking
</syntaxhighlight>

=== Branch Strategy ===
<syntaxhighlight lang="text">
Naming Conventions:
- feature/{description}  - New features
- fix/{issue-number}     - Bug fixes
- add-{issue-number}     - Additions (this repo)
- deps/{package}         - Dependency updates

Include issue numbers for traceability:
- Branch: add-42
- PR body: "Closes #42"
</syntaxhighlight>

=== Issue Linking ===
<syntaxhighlight lang="markdown">
# Keywords that auto-close issues when PR merges:
Closes #42
Fixes #42
Resolves #42

# Multiple issues:
Closes #42, closes #43

# Cross-repository:
Closes owner/repo#42
</syntaxhighlight>

=== Token Requirements ===
<syntaxhighlight lang="text">
GITHUB_TOKEN (default):
- Can create branches and PRs
- PRs won't trigger other workflows
- Limited permissions

Personal Access Token (PAT):
- Full permissions
- PRs can trigger workflows
- Required for workflow chaining
</syntaxhighlight>

=== GitHub Actions Integration ===
<syntaxhighlight lang="yaml">
# Using peter-evans/create-pull-request action
- uses: peter-evans/create-pull-request@v6
  with:
    token: ${{ secrets.PAT }}
    commit-message: "Add feature X"
    title: "Add feature X"
    body: |
      Description here.
      Closes #42
    branch: feature/x
    base: main
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:0PandaDEV_Awesome_windows_create_pull_request_action]]
