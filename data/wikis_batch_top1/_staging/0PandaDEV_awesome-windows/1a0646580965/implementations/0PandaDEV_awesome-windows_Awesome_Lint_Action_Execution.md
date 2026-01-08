# Implementation: Awesome Lint Action Execution

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Repo|awesome-lint-action|https://github.com/Scrum/awesome-lint-action]]
* [[source::Repo|awesome-lint|https://github.com/sindresorhus/awesome-lint]]
|-
! Domains
| [[domain::CI_CD]], [[domain::GitHub_Actions]], [[domain::Linting]]
|-
! Last Updated
| [[last_updated::2026-01-08 13:00 GMT]]
|}

== Overview ==

External tool documentation for GitHub Actions workflow that validates awesome-list format compliance on pull requests.

=== Description ===

This is an '''External Tool Doc''' describing how awesome-windows uses the `Scrum/awesome-lint-action` GitHub Action to validate README.md format on every pull request. The action runs the `awesome-lint` tool and reports results as PR status checks.

=== Usage ===

This workflow runs automatically on all pull requests. No manual invocation is needed. Contributors can view lint results in the PR's "Checks" tab.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/0PandaDEV/awesome-windows awesome-windows]
* '''File:''' .github/workflows/awesome-lint.yml:L1-15

=== External References ===
* [https://github.com/Scrum/awesome-lint-action Scrum/awesome-lint-action GitHub Action]
* [https://github.com/sindresorhus/awesome-lint awesome-lint CLI Tool]

=== Workflow Definition ===

<syntaxhighlight lang="yaml">
name: Awesome readme lint

on:
  pull_request:

jobs:
  build:
    name: awesome readme lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: Scrum/awesome-lint-action@v0.1.1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
</syntaxhighlight>

=== Component Breakdown ===

{| class="wikitable"
|-
! Component !! Version !! Purpose
|-
| actions/checkout || v2 || Clone repository to runner
|-
| Scrum/awesome-lint-action || v0.1.1 || Run awesome-lint validation
|-
| GITHUB_TOKEN || secrets || Authenticate for PR status updates
|}

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| Pull Request Event || GitHub webhook || Yes || Triggers on `pull_request` event
|-
| GITHUB_TOKEN || Secret || Yes || For PR status check updates
|-
| Repository files || Files || Yes || README.md and repo structure
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Status Check || Pass/Fail || PR check status visible in GitHub UI
|-
| Lint Report || Logs || Detailed lint output in Actions logs
|-
| Annotations || GitHub UI || Inline annotations for lint errors
|}

== Usage Examples ==

=== Example 1: Successful Lint ===
<syntaxhighlight lang="text">
Actions tab shows:
✅ awesome readme lint
   All checks passed

PR Status:
✅ All checks have passed
   1 successful check
</syntaxhighlight>

=== Example 2: Failed Lint ===
<syntaxhighlight lang="text">
Actions tab shows:
❌ awesome readme lint
   1 error found

Actions Log:
Running awesome-lint...
✖ List item "zzzApp" is not alphabetically sorted
  Found at line 245

PR Status:
❌ Some checks were not successful
   1 failing check
</syntaxhighlight>

=== Example 3: Viewing Lint Details ===
<syntaxhighlight lang="text">
1. Open Pull Request
2. Click "Details" next to "awesome readme lint" check
3. View full awesome-lint output in Actions logs:

awesome-lint v0.11.0
Checking /home/runner/work/awesome-windows/awesome-windows/README.md

Rules checked:
  ✓ has-awesome-badge
  ✓ has-toc
  ✓ has-contributing
  ✓ items-alphabetical
  ✓ items-have-description
  ✓ links-not-dead

All rules passed!
</syntaxhighlight>

=== Example 4: Running Locally to Debug ===
<syntaxhighlight lang="bash">
# Install awesome-lint
npm install -g awesome-lint

# Run against your changes
awesome-lint README.md

# Fix any issues, then commit and push
git add README.md
git commit --amend
git push -f
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:0PandaDEV_awesome-windows_Awesome_Lint_Validation]]

=== Requires Environment ===
* [[requires_env::Environment:0PandaDEV_awesome-windows_GitHub_Actions_Environment]]
