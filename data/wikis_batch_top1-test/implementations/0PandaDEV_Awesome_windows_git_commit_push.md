# Implementation: Git_commit_push

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Awesome Windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|Git Documentation|https://git-scm.com/docs]]
* [[source::Doc|GitHub Actions Checkout|https://github.com/actions/checkout]]
|-
! Domains
| [[domain::Git]], [[domain::CI_CD]], [[domain::GitHub_Actions]]
|-
! Last Updated
| [[last_updated::2026-01-08 18:30 GMT]]
|}

== Overview ==

Concrete tool for automated git commit and push operations within GitHub Actions workflows.

=== Description ===

This GitHub Actions workflow step performs a conditional git commit and push operation. It configures the git user identity (email and name) for the commit author, stages the modified README.md file, commits with a standard message, and pushes to the remote repository. The step only executes when the previous Python script outputs "Contributors updated".

=== Usage ===

Use this pattern in GitHub Actions workflows when you need to commit and push changes made by previous automation steps. The conditional execution prevents empty commits when no changes were made.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/0PandaDEV/awesome-windows awesome-windows]
* '''File:''' .github/workflows/update_contributors.yml
* '''Lines:''' L34-40

=== Signature ===
<syntaxhighlight lang="yaml">
- name: Commit and push if changed
  if: steps.update.outputs.update_status == 'Contributors updated'
  run: |
    git config --local user.email "70103896+0PandaDEV@users.noreply.github.com"
    git config --local user.name "0PandaDEV"
    git add README.md
    git commit -m "Update contributors" && git push
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="yaml">
# Requires checkout with PAT token for push permissions
- uses: actions/checkout@v4
  with:
    token: ${{ secrets.PAT }}
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| steps.update.outputs.update_status || str || Yes || Must equal "Contributors updated" to trigger
|-
| secrets.PAT || str || Yes || Personal Access Token with repo write permissions
|-
| README.md || file || Yes || Modified file to commit
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Git commit || commit || New commit on main branch with message "Update contributors"
|-
| Remote push || action || Changes pushed to origin/main
|}

== Usage Examples ==

=== Complete Workflow Context ===
<syntaxhighlight lang="yaml">
jobs:
  update-contributors:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          token: ${{ secrets.PAT }}  # Required for push

      - name: Update contributors
        id: update
        run: |
          output=$(python .github/scripts/update_contributors.py)
          echo "update_status=$output" >> $GITHUB_OUTPUT

      - name: Commit and push if changed
        if: steps.update.outputs.update_status == 'Contributors updated'
        run: |
          git config --local user.email "70103896+0PandaDEV@users.noreply.github.com"
          git config --local user.name "0PandaDEV"
          git add README.md
          git commit -m "Update contributors" && git push
</syntaxhighlight>

=== Git Commands Breakdown ===
<syntaxhighlight lang="bash">
# Set commit author identity (local to this repo only)
git config --local user.email "70103896+0PandaDEV@users.noreply.github.com"
git config --local user.name "0PandaDEV"

# Stage specific file
git add README.md

# Commit with message AND push (chained with &&)
# If commit fails (no changes), push is skipped
git commit -m "Update contributors" && git push
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:0PandaDEV_Awesome_windows_Git_Commit_Automation]]

=== Requires Environment ===
* [[requires_env::Environment:0PandaDEV_Awesome_windows_GitHub_Actions_Ubuntu]]

=== Uses Heuristic ===
* [[uses_heuristic::Heuristic:0PandaDEV_Awesome_windows_Conditional_Git_Commit]]
