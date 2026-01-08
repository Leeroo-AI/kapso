# Implementation: Git Config Add Commit Push

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|Git Config|https://git-scm.com/docs/git-config]]
* [[source::Doc|GitHub Actions|https://docs.github.com/en/actions]]
|-
! Domains
| [[domain::Git]], [[domain::CI_CD]], [[domain::GitHub_Actions]]
|-
! Last Updated
| [[last_updated::2026-01-08 13:00 GMT]]
|}

== Overview ==

External tool documentation for Git CLI commands used to configure identity, stage, commit, and push changes in GitHub Actions.

=== Description ===

This is an '''External Tool Doc''' describing the Git CLI commands used in the contributor update workflow's final step. It configures the commit author identity, stages README.md changes, creates a commit, and pushes to the main branch.

=== Usage ===

This step executes conditionally—only when the Python script outputs "Contributors updated". It runs after the README has been modified by the Python script.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/0PandaDEV/awesome-windows awesome-windows]
* '''File:''' .github/workflows/update_contributors.yml:L34-40

=== External Reference ===
* [https://git-scm.com/docs Git Documentation]
* [https://docs.github.com/en/actions GitHub Actions Documentation]

=== Workflow Step Definition ===

<syntaxhighlight lang="yaml">
- name: Commit and push if changed
  if: steps.update.outputs.update_status == 'Contributors updated'
  run: |
    git config --local user.email "70103896+0PandaDEV@users.noreply.github.com"
    git config --local user.name "0PandaDEV"
    git add README.md
    git commit -m "Update contributors" && git push
</syntaxhighlight>

=== Command Breakdown ===

{| class="wikitable"
|-
! Command !! Purpose
|-
| `git config --local user.email "..."` || Set commit author email (local scope)
|-
| `git config --local user.name "..."` || Set commit author name (local scope)
|-
| `git add README.md` || Stage the modified file
|-
| `git commit -m "..."` || Create commit with message
|-
| `git push` || Push to remote (origin main)
|}

=== Conditional Execution ===

<syntaxhighlight lang="yaml">
# Only runs when Python script outputs "Contributors updated"
if: steps.update.outputs.update_status == 'Contributors updated'

# The Python script outputs one of:
# - "Contributors updated" → triggers commit
# - "No changes in contributors" → skips commit
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| update_status || String || Yes || Output from Python script step
|-
| README.md || File || Yes || Modified file to commit
|-
| secrets.PAT || Token || Yes || Token for git push authentication (checkout step)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Git commit || Commit || "Update contributors" commit on main branch
|-
| Remote push || Action || Changes pushed to origin/main
|}

== Usage Examples ==

=== Example 1: Successful Update ===
<syntaxhighlight lang="text">
Python script output: "Contributors updated"

Step executes:
$ git config --local user.email "70103896+0PandaDEV@users.noreply.github.com"
$ git config --local user.name "0PandaDEV"
$ git add README.md
$ git commit -m "Update contributors"
[main abc1234] Update contributors
 1 file changed, 5 insertions(+), 3 deletions(-)
$ git push
To github.com:0PandaDEV/awesome-windows.git
   def5678..abc1234  main -> main
</syntaxhighlight>

=== Example 2: No Changes (Step Skipped) ===
<syntaxhighlight lang="text">
Python script output: "No changes in contributors"

Step condition: steps.update.outputs.update_status == 'Contributors updated'
Condition result: 'No changes in contributors' != 'Contributors updated'
→ Step skipped

Actions log shows:
✓ Commit and push if changed
  Skipped (condition not met)
</syntaxhighlight>

=== Example 3: Commit Attribution ===
<syntaxhighlight lang="text">
# Git log shows the automated commit:
$ git log --oneline -1
abc1234 Update contributors

$ git log -1 --format=fuller
commit abc1234...
Author:     0PandaDEV <70103896+0PandaDEV@users.noreply.github.com>
AuthorDate: Wed Jan 8 00:00:05 2026 +0000
Commit:     0PandaDEV <70103896+0PandaDEV@users.noreply.github.com>
CommitDate: Wed Jan 8 00:00:05 2026 +0000

    Update contributors
</syntaxhighlight>

=== Example 4: Error Handling with && ===
<syntaxhighlight lang="bash">
# The command uses && to chain commit and push:
git commit -m "Update contributors" && git push

# If commit fails (e.g., no changes despite condition passing):
# - Exit code is non-zero
# - git push is NOT executed
# - Step fails (but this is unlikely given the condition)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:0PandaDEV_awesome-windows_Git_Commit_Automation]]

=== Requires Environment ===
* [[requires_env::Environment:0PandaDEV_awesome-windows_GitHub_Actions_Environment]]
