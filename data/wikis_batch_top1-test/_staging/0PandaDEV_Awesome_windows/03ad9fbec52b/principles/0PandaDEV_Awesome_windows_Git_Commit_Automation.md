# Principle: Git_Commit_Automation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Git Documentation|https://git-scm.com/docs]]
* [[source::Doc|GitHub Actions Checkout|https://github.com/actions/checkout]]
|-
! Domains
| [[domain::Git]], [[domain::CI_CD]], [[domain::Version_Control]]
|-
! Last Updated
| [[last_updated::2026-01-08 18:30 GMT]]
|}

== Overview ==

Technique for programmatically committing and pushing changes to Git repositories in CI/CD pipelines.

=== Description ===

Git Commit Automation enables CI/CD workflows to commit and push changes made during automated processes. This includes setting up git identity (author information), staging modified files, creating commits with appropriate messages, and pushing to remote repositories. Proper implementation requires handling authentication, conditional execution, and avoiding empty commits.

Key considerations include using tokens with appropriate permissions, configuring the git user identity for the runner environment, and implementing guards to prevent commits when no changes exist.

=== Usage ===

Use this principle when:
- Automating file updates that should be version controlled
- Creating automated changelog or documentation updates
- Implementing self-updating repositories
- Building bots that propose changes via commits

== Theoretical Basis ==

=== Git Operations Sequence ===
<syntaxhighlight lang="text">
1. Configure Identity
   git config --local user.email "..."
   git config --local user.name "..."

2. Stage Changes
   git add <files>

3. Create Commit
   git commit -m "message"

4. Push to Remote
   git push
</syntaxhighlight>

=== Authentication Methods ===
{| class="wikitable"
|-
! Method !! Setup !! Permissions
|-
| GITHUB_TOKEN || Automatic || Limited (can't trigger workflows)
|-
| Personal Access Token || Manual secret || Full (can trigger workflows)
|-
| Deploy Key || Repository settings || Repository-specific
|}

=== Conditional Commit Pattern ===
<syntaxhighlight lang="bash">
# Chain with && to skip push if commit fails (no changes)
git commit -m "message" && git push

# Or check for changes first
if git diff --quiet; then
  echo "No changes to commit"
else
  git add . && git commit -m "message" && git push
fi
</syntaxhighlight>

=== Identity Configuration ===
<syntaxhighlight lang="text">
For GitHub-linked commits, use:
- Email: {user_id}+{username}@users.noreply.github.com
- Name: GitHub username

This links commits to GitHub profiles while using the
noreply email for privacy.
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:0PandaDEV_Awesome_windows_git_commit_push]]
