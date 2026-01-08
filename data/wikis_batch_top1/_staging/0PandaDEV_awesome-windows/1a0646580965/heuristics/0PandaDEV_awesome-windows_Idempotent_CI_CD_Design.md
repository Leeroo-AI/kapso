# Idempotent_CI_CD_Design

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Experience|Internal|CI/CD best practices]]
|-
! Domains
| [[domain::CI_CD]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-01-08 00:00 GMT]]
|}

== Overview ==

CI/CD workflows should detect changes before committing to avoid empty commits and unnecessary Git history pollution.

=== Description ===

An idempotent CI/CD design ensures that running the same workflow multiple times produces the same result without side effects. For automated README updates, this means checking if contributors have actually changed before attempting to update the file and create a commit. This pattern prevents empty commits, reduces Git history noise, and saves CI resources.

=== Usage ===

Use this heuristic when **designing automated update workflows** that modify repository files on a schedule. Always implement a change detection step before the commit step.

== The Insight (Rule of Thumb) ==

* **Action:** Implement a `has_changed()` check before creating commits.
* **Value:** Only commit when actual changes are detected.
* **Trade-off:** Adds complexity to the workflow; requires state comparison logic.
* **Pattern:** `if has_changes: commit()` rather than unconditional `commit()`.

== Reasoning ==

Idempotent design provides:
1. **Clean Git history:** No empty or duplicate commits
2. **Resource efficiency:** Skip unnecessary file writes and Git operations
3. **Auditability:** Each commit represents a meaningful change
4. **Reduced noise:** CI logs are cleaner; notifications only fire on actual updates

== Code Evidence ==

Change detection function from `.github/scripts/update_contributors.py:14-22`:
<syntaxhighlight lang="python">
def has_contributors_changed(contributors):
    with open('README.md', 'r') as file:
        content = file.read()

    for contributor in contributors:
        username = contributor['login']
        if f"https://github.com/{username}" not in content:
            return True
    return False
</syntaxhighlight>

Conditional execution in main block from `.github/scripts/update_contributors.py:44-50`:
<syntaxhighlight lang="python">
if __name__ == "__main__":
    contributors = get_contributors()
    if has_contributors_changed(contributors):
        update_readme(contributors)
        print("Contributors updated")
    else:
        print("No changes in contributors")
</syntaxhighlight>

Conditional commit in workflow from `.github/workflows/update_contributors.yml:34-35`:
<syntaxhighlight lang="yaml">
      - name: Commit and push if changed
        if: steps.update.outputs.update_status == 'Contributors updated'
</syntaxhighlight>

== Related Pages ==

=== Used By ===
This heuristic is referenced by:
* Implementation: has_contributors_changed_Function
* Implementation: Git_Config_Add_Commit_Push
* Workflow: Automated_Contributor_Update
