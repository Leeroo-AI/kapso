# Principle: Comment_Command_Trigger

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|GitHub Actions Events|https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows]]
* [[source::Blog|ChatOps Pattern|https://www.atlassian.com/blog/software-teams/what-is-chatops-adoption-guide]]
|-
! Domains
| [[domain::GitHub_Actions]], [[domain::ChatOps]], [[domain::Event_Driven]]
|-
! Last Updated
| [[last_updated::2026-01-08 18:30 GMT]]
|}

== Overview ==

Technique for triggering automated workflows via specific comment commands on issues or pull requests.

=== Description ===

Comment Command Triggers implement the ChatOps pattern in GitHub repositories. By listening for `issue_comment` events, workflows can detect specific command strings (like `/convert`, `/deploy`, `/test`) and execute corresponding automation. This provides a conversational interface for repository operations without requiring CLI access or direct workflow dispatch.

Key aspects include command parsing, authorization checks (who can issue commands), and context validation (what types of issues/PRs support which commands).

=== Usage ===

Use this principle when:
- Implementing maintainer-only automation triggers
- Building conversational bot interfaces
- Creating manual approval gates in automated workflows
- Providing intuitive automation interfaces for non-technical users

== Theoretical Basis ==

=== Event Flow ===
<syntaxhighlight lang="text">
1. User posts comment on issue/PR
2. GitHub emits issue_comment event
3. Workflow receives event with comment data:
   - comment.body (the text)
   - comment.user.login (who posted)
   - issue context (labels, state)
4. Workflow evaluates conditions
5. If conditions match, job executes
</syntaxhighlight>

=== Authorization Pattern ===
<syntaxhighlight lang="yaml">
# Multi-condition check for secure command handling
if: |
  github.event.comment.body == '/command' &&      # Exact command
  github.event.comment.user.login == 'maintainer' &&  # Authorized user
  contains(github.event.issue.labels.*.name, 'ready')  # Context check
</syntaxhighlight>

=== Common Commands ===
{| class="wikitable"
|-
! Command !! Purpose !! Example Use
|-
| /approve || Approve changes || Code review bots
|-
| /deploy || Trigger deployment || Staging/production deploys
|-
| /retry || Re-run failed jobs || CI recovery
|-
| /convert || Transform issue to PR || This repository
|}

=== Security Considerations ===
<syntaxhighlight lang="text">
1. Always verify commenter identity
   - Check against allowlist of users
   - Or check organization membership

2. Validate issue/PR context
   - Correct labels present
   - Correct state (open, not locked)

3. Use conditions, not code
   - GitHub Actions `if:` runs before checkout
   - Prevents unauthorized code execution
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:0PandaDEV_Awesome_windows_convert_command_check]]
