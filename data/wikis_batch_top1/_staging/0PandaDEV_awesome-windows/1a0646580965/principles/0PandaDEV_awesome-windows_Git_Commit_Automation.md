# Principle: Git Commit Automation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|Git Documentation|https://git-scm.com/docs]]
* [[source::Doc|GitHub Actions Git|https://docs.github.com/en/actions/using-workflows/using-github-cli-in-workflows]]
|-
! Domains
| [[domain::Git]], [[domain::CI_CD]], [[domain::Automation]]
|-
! Last Updated
| [[last_updated::2026-01-08 13:00 GMT]]
|}

== Overview ==

Principle for automating Git operations (configure, add, commit, push) within CI/CD workflows.

=== Description ===

Git Commit Automation enables CI/CD pipelines to commit and push changes without manual intervention. In GitHub Actions, this requires:

- '''Identity configuration:''' Setting user.name and user.email
- '''Staging changes:''' `git add` for modified files
- '''Committing:''' Creating a commit with message
- '''Pushing:''' Pushing to remote repository
- '''Conditional execution:''' Only run if changes exist

This principle ensures automated changes are properly attributed and pushed to the repository.

=== Usage ===

Apply this principle when CI/CD workflows need to:
- Update generated files
- Sync documentation
- Auto-format code
- Update version files
- Maintain automated content

== Theoretical Basis ==

'''Git Identity in CI:'''

Git requires user identity for commits. In GitHub Actions, this is not automatically set, so workflows must configure it:

<syntaxhighlight lang="bash">
git config --local user.email "user@example.com"
git config --local user.name "User Name"
</syntaxhighlight>

Using `--local` ensures configuration is scoped to the repository, not the global system.

'''Conditional Commit Pattern:'''

<syntaxhighlight lang="text">
Check for changes
      │
      ├── No changes → Skip commit
      │
      └── Has changes → git add → git commit → git push
</syntaxhighlight>

'''Methods to Check for Changes:'''

{| class="wikitable"
|-
! Method !! Command !! Use Case
|-
| Script output || Custom flag in script || When script tracks changes
|-
| Git status || `git diff --quiet` || General file changes
|-
| Git diff || `git diff --name-only` || Specific file tracking
|}

'''Authentication:'''

GitHub Actions provides `GITHUB_TOKEN` for authentication, but for pushing to protected branches or triggering workflows, a Personal Access Token (PAT) may be required.

== Practical Guide ==

=== Standard Commit Flow ===

<syntaxhighlight lang="bash">
# 1. Configure identity
git config --local user.email "email@example.com"
git config --local user.name "Bot Name"

# 2. Stage changes
git add README.md

# 3. Commit (fail silently if no changes)
git commit -m "Update content" || echo "No changes"

# 4. Push
git push
</syntaxhighlight>

=== Conditional Execution ===

Only commit if previous step indicated changes:

<syntaxhighlight lang="yaml">
- name: Update content
  id: update
  run: |
    output=$(python script.py)
    echo "status=$output" >> $GITHUB_OUTPUT

- name: Commit changes
  if: steps.update.outputs.status == 'Updated'
  run: |
    git config --local user.email "..."
    git commit -m "..." && git push
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:0PandaDEV_awesome-windows_Git_Config_Add_Commit_Push]]

=== Part of Workflow ===
* [[workflow::Workflow:0PandaDEV_awesome-windows_Automated_Contributor_Update]]
