# Implementation: Git Fork Edit Workflow

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|GitHub Fork Workflow|https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork]]
|-
! Domains
| [[domain::Git]], [[domain::Version_Control]], [[domain::Community_Contribution]]
|-
! Last Updated
| [[last_updated::2026-01-08 13:00 GMT]]
|}

== Overview ==

Pattern documentation for the fork-edit-commit-PR workflow used for manual awesome-list contributions.

=== Description ===

This is a '''Pattern Doc''' describing the standard GitHub fork-based contribution workflow as applied to awesome-windows. The process involves forking the repository, editing README.md to add an entry, and submitting a pull request.

=== Usage ===

Use this workflow pattern when submitting applications via the Manual PR path (Step 4). Requires Git/GitHub familiarity and adherence to formatting conventions.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/0PandaDEV/awesome-windows awesome-windows]
* '''File:''' CONTRIBUTING.md:L32-37

=== Documented Workflow ===

From CONTRIBUTING.md:

<syntaxhighlight lang="markdown">
If you don't have a [GitHub account](https://github.com/join), make one!

1. Fork this repo.
2. Make changes under correct section in `README.md`
3. Update `Contents` (if applicable)
4. Commit and open a Pull Request
</syntaxhighlight>

=== Interface Specification ===

<syntaxhighlight lang="bash">
# Step 1: Fork via GitHub UI or CLI
gh repo fork 0PandaDEV/awesome-windows --clone

# Step 2: Edit README.md
# Add entry in format: * [App Name](url) - Description [icons]

# Step 3: Commit changes
git add README.md
git commit -m "Add [App Name] to [Category] category"

# Step 4: Push to fork
git push origin main

# Step 5: Create pull request
gh pr create --title "Add [App Name] to [Category] category" \
             --body "Application URL: https://...
Repository: https://github.com/...
Description: ..."
</syntaxhighlight>

=== Entry Format Template ===

<syntaxhighlight lang="markdown">
# Standard entry
* [App Name](https://official-url.com) - Brief description.

# Open source entry
* [App Name](https://official-url.com) - Brief description. [![Open-Source Software][oss]](https://github.com/repo)

# Paid entry
* [App Name](https://official-url.com) - Brief description. ![paid]

# Open source + paid entry
* [App Name](https://official-url.com) - Brief description. [![Open-Source Software][oss]](https://github.com/repo) ![paid]
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| Application metadata || Dict || Yes || Name, URL, category, description from Step 1
|-
| Repository access || GitHub account || Yes || Ability to fork and create PRs
|-
| Git installation || CLI tool || Yes || Git CLI for local development
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Forked repository || GitHub repo || Your copy of awesome-windows
|-
| Modified README.md || File || With new entry added alphabetically
|-
| Pull Request || GitHub PR || Request to merge changes to upstream
|}

== Usage Examples ==

=== Example 1: Adding VS Code (OSS) ===
<syntaxhighlight lang="bash">
# Clone your fork
git clone https://github.com/YOUR_USERNAME/awesome-windows.git
cd awesome-windows

# Edit README.md - add under ## IDEs section (alphabetically)
# Find: ## IDEs
# Add after appropriate alphabetical position:
# * [Visual Studio Code](https://code.visualstudio.com/) - Lightweight but powerful source code editor. [![Open-Source Software][oss]](https://github.com/microsoft/vscode)

# Commit
git add README.md
git commit -m "Add Visual Studio Code to IDEs category"

# Push and create PR
git push origin main
gh pr create --title "Add Visual Studio Code to IDEs category"
</syntaxhighlight>

=== Example 2: Adding Paid Application ===
<syntaxhighlight lang="bash">
# Edit README.md - add under ## Text Editors section
# * [Sublime Text](https://www.sublimetext.com/) - Sophisticated text editor for code. ![paid]

git add README.md
git commit -m "Add Sublime Text to Text Editors category"
git push origin main
</syntaxhighlight>

=== Example 3: Batch Submission (Multiple Apps) ===
<syntaxhighlight lang="bash">
# Add multiple entries in a single PR
# Edit README.md with all entries in their respective categories

git add README.md
git commit -m "Add multiple developer tools

- Add VS Code to IDEs
- Add Windows Terminal to Terminal
- Add WinMerge to Developer Utilities"

git push origin main
gh pr create --title "Add multiple developer tools" \
             --body "This PR adds:
- VS Code (IDEs)
- Windows Terminal (Terminal)
- WinMerge (Developer Utilities)"
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:0PandaDEV_awesome-windows_Manual_PR_Submission]]

=== Requires Environment ===
* [[requires_env::Environment:0PandaDEV_awesome-windows_Local_Git_Environment]]
