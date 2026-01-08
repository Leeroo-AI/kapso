# Heuristic: Conditional_Git_Commit

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|0PandaDEV/awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|GitHub Actions Conditional Steps|https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions#jobsjob_idstepsif]]
|-
! Domains
| [[domain::CI_CD]], [[domain::Git_Operations]]
|-
! Last Updated
| [[last_updated::2026-01-08 18:30 GMT]]
|}

== Overview ==
Chain git commands with `&&` to only push when commit succeeds, preventing empty commit errors.

=== Description ===
This heuristic addresses a common CI/CD problem: git operations that fail when there are no changes to commit. By chaining `git commit` and `git push` with `&&`, the push only executes if the commit succeeds. Combined with GitHub Actions `if` conditions, this creates a robust pattern for conditional git operations.

=== Usage ===
Use this heuristic when you have **automated workflows** that may or may not produce file changes, and you need to:
- Avoid "nothing to commit" errors
- Only push when there are actual changes
- Chain dependent git operations safely

== The Insight (Rule of Thumb) ==

* **Action:** Chain git operations with `&&` operator: `git commit -m "message" && git push`
* **Value:** The push command only runs if commit exits with status 0 (success)
* **Trade-off:** If commit fails for any reason (not just "nothing to commit"), push is skipped
* **Complement:** Use GitHub Actions `if:` condition to skip the entire step when no changes detected

== Reasoning ==

The `&&` operator in bash is a logical AND that short-circuits:
- If `git commit` succeeds (exit 0), `git push` runs
- If `git commit` fails (exit 1, e.g., "nothing to commit"), `git push` is skipped

This is preferable to:
1. **Separate steps:** Would fail the workflow on empty commit
2. **Using `; ` separator:** Would run push even after failed commit
3. **Using `||`:** Would run push on commit failure (opposite of desired)

The pattern combines three layers of protection:
1. **Script output check:** Python script outputs status string
2. **Step condition:** `if: steps.update.outputs.update_status == 'Contributors updated'`
3. **Command chaining:** `git commit && git push`

== Code Evidence ==

Conditional git operations from `update_contributors.yml:34-40`:
<syntaxhighlight lang="yaml">
- name: Commit and push if changed
  if: steps.update.outputs.update_status == 'Contributors updated'
  run: |
    git config --local user.email "70103896+0PandaDEV@users.noreply.github.com"
    git config --local user.name "0PandaDEV"
    git add README.md
    git commit -m "Update contributors" && git push
</syntaxhighlight>

=== Three-Layer Protection Pattern ===

<syntaxhighlight lang="text">
Layer 1: Script-level detection
  └── Python outputs "Contributors updated" or "No changes in contributors"

Layer 2: Workflow step condition
  └── if: steps.update.outputs.update_status == 'Contributors updated'
  └── Skips entire step if no changes detected

Layer 3: Command-level safety
  └── git commit -m "..." && git push
  └── Push only runs if commit succeeds
</syntaxhighlight>

=== Alternative Patterns (Less Robust) ===

<syntaxhighlight lang="bash">
# BAD: Separate commands - fails workflow on empty commit
git commit -m "Update"
git push

# BAD: Semicolon - pushes even after failed commit
git commit -m "Update"; git push

# BAD: Force commit - creates empty commits
git commit --allow-empty -m "Update" && git push
</syntaxhighlight>

== Related Pages ==

* [[used_by::Implementation:0PandaDEV_Awesome_windows_git_commit_push]]
