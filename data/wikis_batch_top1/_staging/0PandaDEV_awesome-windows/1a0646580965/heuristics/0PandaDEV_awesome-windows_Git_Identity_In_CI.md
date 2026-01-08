# Git_Identity_In_CI

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|GitHub Actions Documentation|https://docs.github.com/en/actions]]
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
|-
! Domains
| [[domain::CI_CD]], [[domain::Git]]
|-
! Last Updated
| [[last_updated::2026-01-08 00:00 GMT]]
|}

== Overview ==

Always configure git user.name and user.email before committing in CI environments to ensure proper commit attribution.

=== Description ===

GitHub Actions runners do not have a default Git identity configured. Before making any commits in a CI workflow, you must explicitly set the `user.name` and `user.email` Git configuration. Using the GitHub noreply email format (`{id}+{username}@users.noreply.github.com`) ensures commits are properly attributed to the account without exposing personal email addresses.

=== Usage ===

Use this heuristic when **automating Git commits in CI/CD workflows**. Always configure identity before `git commit` commands.

== The Insight (Rule of Thumb) ==

* **Action:** Run `git config --local user.email` and `git config --local user.name` before committing.
* **Value:** Use GitHub's noreply email format: `{user_id}+{username}@users.noreply.github.com`.
* **Trade-off:** Requires knowing the GitHub user ID; use `--local` to avoid polluting global config.
* **Scope:** Use `--local` flag to scope config to the repository only.

== Reasoning ==

Proper Git identity configuration:
1. **Attribution:** Commits show the correct author in Git history
2. **Privacy:** Noreply email protects personal email address
3. **GitHub integration:** Commits link to the correct GitHub profile
4. **Compliance:** Some repositories require verified commits

== Code Evidence ==

Git identity configuration from `.github/workflows/update_contributors.yml:37-38`:
<syntaxhighlight lang="yaml">
          git config --local user.email "70103896+0PandaDEV@users.noreply.github.com"
          git config --local user.name "0PandaDEV"
</syntaxhighlight>

== Related Pages ==

=== Used By ===
This heuristic is referenced by:
* Implementation: Git_Config_Add_Commit_Push
* Workflow: Automated_Contributor_Update
