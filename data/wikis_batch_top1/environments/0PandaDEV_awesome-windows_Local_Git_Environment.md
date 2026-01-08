# Local_Git_Environment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Git Documentation|https://git-scm.com/doc]]
* [[source::Doc|GitHub Fork Workflow|https://docs.github.com/en/get-started/quickstart/fork-a-repo]]
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Version_Control]]
|-
! Last Updated
| [[last_updated::2026-01-08 00:00 GMT]]
|}

== Overview ==

Local development environment with Git CLI for manual fork-edit-PR contribution workflow.

=== Description ===

This environment provides a local Git installation for contributors who prefer the manual pull request workflow. It enables forking the repository, making local edits to README.md, committing changes, and creating pull requests via the command line or Git GUI clients.

=== Usage ===

Use this environment for any **manual PR-based contribution** workflow that bypasses the GitHub Issue Forms interface. This is the mandatory prerequisite for running the `Git_Fork_Edit_Workflow` implementation.

== System Requirements ==

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Windows, macOS, or Linux || Any Git-supported platform
|-
| Git || Git 2.x+ || `git --version` to verify
|-
| Network || Internet connection || Required for push/pull operations
|-
| Editor || Any text editor || For editing README.md
|}

== Dependencies ==

=== System Packages ===
* `git` >= 2.0

=== Optional ===
* Git GUI client (GitHub Desktop, GitKraken, SourceTree, etc.)
* SSH key configured for GitHub authentication

== Credentials ==

The following are required for authenticated Git operations:
* `GitHub Account`: Required for forking and creating pull requests
* `SSH Key` or `HTTPS Token`: For pushing to forked repository

== Quick Install ==

<syntaxhighlight lang="bash">
# Windows (via winget)
winget install Git.Git

# macOS (via Homebrew)
brew install git

# Linux (Ubuntu/Debian)
sudo apt-get install git

# Verify installation
git --version
</syntaxhighlight>

== Code Evidence ==

Contribution guidelines referencing Git workflow from `CONTRIBUTING.md:32-37`:
<syntaxhighlight lang="markdown">
If you don't have a [GitHub account](https://github.com/join), make one!

1. Fork this repo.
2. Make changes under correct section in `README.md`
3. Update `Contents` (if applicable)
4. Commit and open a Pull Request
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `Permission denied (publickey)` || SSH key not configured || Add SSH key to GitHub account or use HTTPS
|-
|| `remote: Repository not found` || Incorrect fork URL || Verify fork exists and URL is correct
|-
|| `Updates were rejected` || Local branch behind remote || Run `git pull --rebase origin main` first
|-
|| `fatal: not a git repository` || Not in repo directory || `cd` to cloned repository directory
|}

== Compatibility Notes ==

* '''Windows:''' Git Bash or PowerShell recommended; WSL also supported
* '''Line endings:''' Configure `core.autocrlf` appropriately for cross-platform compatibility
* '''Credentials:''' Windows users may need Git Credential Manager for HTTPS authentication

== Related Pages ==

=== Required By ===
This environment is required by:
* Implementation: Git_Fork_Edit_Workflow
