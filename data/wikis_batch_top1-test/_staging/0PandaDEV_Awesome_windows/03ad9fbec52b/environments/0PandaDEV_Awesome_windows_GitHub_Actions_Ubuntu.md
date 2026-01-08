# Environment: GitHub_Actions_Ubuntu

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|0PandaDEV/awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|GitHub Actions Runners|https://docs.github.com/en/actions/using-github-hosted-runners]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::CI_CD]]
|-
! Last Updated
| [[last_updated::2026-01-08 18:30 GMT]]
|}

== Overview ==
GitHub Actions ubuntu-latest runner with git CLI, awk, and shell scripting capabilities.

=== Description ===
This environment provides a standard GitHub-hosted Ubuntu runner for CI/CD workflows. It includes pre-installed tools like `git`, `awk`, `bash`, and `grep` that are essential for file manipulation, text processing, and version control operations. The runner uses the `ubuntu-latest` image which includes a comprehensive set of development tools.

=== Usage ===
Use this environment for any **shell-based automation**, **README manipulation**, or **git operations** in GitHub Actions. It is the mandatory prerequisite for running workflows that parse issue bodies, modify files with awk scripts, and commit changes back to the repository.

== System Requirements ==

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Ubuntu (latest) || GitHub-hosted runner with pre-installed tools
|-
| Shell || Bash || Default shell for run steps
|-
| Network || Outbound HTTPS || Required for git push operations
|}

== Dependencies ==

=== System Packages ===
Pre-installed on ubuntu-latest:
* `git` - Version control CLI
* `awk` (GNU awk) - Text processing
* `bash` - Shell interpreter
* `grep` - Pattern matching
* `xargs` - Argument builder

=== GitHub Actions ===
* `actions/checkout@v4` - Repository checkout
* `actions/github-script@v7` - GitHub API access from JavaScript
* `peter-evans/create-pull-request@v6` - Automated PR creation

== Credentials ==

The following secrets must be configured:
* `PAT`: GitHub Personal Access Token with `repo` scope for:
  - Pushing commits to branches
  - Creating pull requests
  - Closing issues via API

== Quick Install ==

<syntaxhighlight lang="bash">
# No additional installation required for ubuntu-latest runner
# All dependencies are pre-installed

# Verify tools are available
which git awk bash grep
</syntaxhighlight>

== Code Evidence ==

Git operations from `update_contributors.yml:34-40`:
<syntaxhighlight lang="yaml">
- name: Commit and push if changed
  if: steps.update.outputs.update_status == 'Contributors updated'
  run: |
    git config --local user.email "70103896+0PandaDEV@users.noreply.github.com"
    git config --local user.name "0PandaDEV"
    git add README.md
    git commit -m "Update contributors" && git push
</syntaxhighlight>

awk text processing from `covert_to_pr.yml:75-99`:
<syntaxhighlight lang="bash">
awk -v new_entry="$NEW_ENTRY" -v category="$CATEGORY" '
BEGIN {in_category=0; added=0}
/^## / {
  if (in_category && !added) {
    print new_entry
    added=1
  }
  in_category = ($0 ~ "^## " category)
  print
  if (in_category) print ""
  next
}
' README.md > README.md.tmp && mv README.md.tmp README.md
</syntaxhighlight>

Shell variable extraction from `covert_to_pr.yml:47-51`:
<syntaxhighlight lang="bash">
APP_NAME=$(echo "$ISSUE_BODY" | awk '/### Application Name/{flag=1; next} /###/{flag=0} flag' | xargs)
APP_URL=$(echo "$ISSUE_BODY" | awk '/### Application URL/{flag=1; next} /###/{flag=0} flag' | xargs)
CATEGORY=$(echo "$ISSUE_BODY" | awk '/### Category/{flag=1; next} /###/{flag=0} flag' | xargs)
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `fatal: could not read Username` || PAT not provided to checkout || Add `token: ${{ secrets.PAT }}` to checkout step
|-
|| `Permission denied` on git push || PAT lacks repo scope || Regenerate PAT with `repo` scope
|-
|| `awk: command not found` || Non-standard runner image || Use `ubuntu-latest` or install gawk manually
|}

== Compatibility Notes ==

* '''ubuntu-latest:''' Image version updates periodically; workflows should be resilient to minor version changes
* '''Self-hosted runners:''' May require manual installation of awk, git if not using GitHub-hosted runners
* '''PAT Permissions:''' Requires `repo` scope; fine-grained tokens need explicit repository access

== Related Pages ==

* [[required_by::Implementation:0PandaDEV_Awesome_windows_git_commit_push]]
* [[required_by::Implementation:0PandaDEV_Awesome_windows_convert_command_check]]
* [[required_by::Implementation:0PandaDEV_Awesome_windows_issue_metadata_extraction]]
* [[required_by::Implementation:0PandaDEV_Awesome_windows_entry_builder]]
* [[required_by::Implementation:0PandaDEV_Awesome_windows_awk_insert_sorted]]
* [[required_by::Implementation:0PandaDEV_Awesome_windows_create_pull_request_action]]
* [[required_by::Implementation:0PandaDEV_Awesome_windows_close_issue_action]]
