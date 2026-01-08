# GitHub_Actions_Environment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|GitHub Actions Documentation|https://docs.github.com/en/actions]]
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::CI_CD]]
|-
! Last Updated
| [[last_updated::2026-01-08 00:00 GMT]]
|}

== Overview ==

GitHub Actions runner environment (ubuntu-latest) for automated CI/CD workflows including linting, PR conversion, and contributor updates.

=== Description ===

This environment provides the GitHub Actions `ubuntu-latest` runner context for executing automated workflows. It includes the GitHub runner software, pre-installed tools (Git, Python, Node.js), and access to repository secrets and the GitHub API via action contexts. The environment supports scheduled cron jobs, webhook triggers, and manual workflow dispatch.

=== Usage ===

Use this environment for any **automated CI/CD workflow** that runs on GitHub Actions. This is the mandatory prerequisite for running the `Awesome_Lint_Action_Execution`, `Issue_To_PR_Conversion`, `GitHub_Actions_Cron_Schedule`, and `Git_Config_Add_Commit_Push` implementations.

== System Requirements ==

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| Runner || ubuntu-latest || GitHub-hosted runner
|-
| Hardware || 2-core CPU, 7GB RAM, 14GB SSD || Standard GitHub-hosted runner specs
|-
| Network || Full internet access || For API calls and package installation
|}

== Dependencies ==

=== Pre-installed on ubuntu-latest ===
* `git` (latest)
* `python3` (multiple versions available)
* `node` (multiple versions available)
* `curl`, `wget`, `jq`

=== GitHub Actions Used ===
* `actions/checkout@v4` - Repository checkout
* `actions/setup-python@v5` - Python environment setup
* `actions/github-script@v7` - GitHub API scripting
* `peter-evans/create-pull-request@v6` - Automated PR creation
* `Scrum/awesome-lint-action@v0.1.1` - Awesome list linting

== Credentials ==

The following environment variables/secrets must be configured:
* `GITHUB_TOKEN`: Automatically provided by GitHub Actions (read/write for repo)
* `PAT` (Personal Access Token): Required for cross-repository actions and PR creation
  * Needed for: checkout with custom token, create-pull-request action

== Quick Install ==

<syntaxhighlight lang="bash">
# No manual installation required
# Environment is automatically provisioned by GitHub Actions
# Secrets must be configured in repository Settings > Secrets and variables > Actions
</syntaxhighlight>

== Code Evidence ==

Runner specification from `.github/workflows/update_contributors.yml:9-10`:
<syntaxhighlight lang="yaml">
  update-contributors:
    runs-on: ubuntu-latest
</syntaxhighlight>

PAT token usage for checkout from `.github/workflows/update_contributors.yml:12-14`:
<syntaxhighlight lang="yaml">
      - uses: actions/checkout@v4
        with:
          token: ${{ secrets.PAT }}
</syntaxhighlight>

Python setup from `.github/workflows/update_contributors.yml:16-19`:
<syntaxhighlight lang="yaml">
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.x"
</syntaxhighlight>

Cron schedule trigger from `.github/workflows/update_contributors.yml:3-6`:
<syntaxhighlight lang="yaml">
on:
  schedule:
    - cron: "0 0 * * *"
  workflow_dispatch:
</syntaxhighlight>

Git identity configuration from `.github/workflows/update_contributors.yml:36-38`:
<syntaxhighlight lang="yaml">
          git config --local user.email "70103896+0PandaDEV@users.noreply.github.com"
          git config --local user.name "0PandaDEV"
          git add README.md
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `Resource not accessible by integration` || GITHUB_TOKEN lacks permissions || Use PAT secret instead of GITHUB_TOKEN
|-
|| `fatal: could not read Password` || PAT not configured || Add PAT to repository secrets
|-
|| `Error: Process completed with exit code 1` || Script/action failure || Check step logs for specific error
|-
|| Workflow not triggering || Cron syntax error or branch protection || Verify cron syntax; check Actions permissions
|}

== Compatibility Notes ==

* '''Secrets:''' PAT must have `repo` scope for write access to repository
* '''Rate limits:''' GitHub API has rate limits; authenticated requests have higher limits
* '''Concurrency:''' Multiple workflow runs may conflict; use `concurrency` key to prevent
* '''Fork PRs:''' Workflows triggered from forks have limited secret access for security

== Related Pages ==

=== Required By ===
This environment is required by:
* Implementation: Awesome_Lint_Action_Execution
* Implementation: Issue_To_PR_Conversion
* Implementation: GitHub_Actions_Cron_Schedule
* Implementation: Git_Config_Add_Commit_Push
