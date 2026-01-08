# Principle: Issue_State_Management

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|GitHub REST API - Issues|https://docs.github.com/en/rest/issues/issues]]
* [[source::Doc|GitHub Script Action|https://github.com/actions/github-script]]
|-
! Domains
| [[domain::GitHub]], [[domain::Issue_Management]], [[domain::Automation]]
|-
! Last Updated
| [[last_updated::2026-01-08 18:30 GMT]]
|}

== Overview ==

Technique for programmatically managing GitHub Issue lifecycle states via the API.

=== Description ===

Issue State Management automates the lifecycle of GitHub Issues, including opening, closing, labeling, and assigning. This enables workflows to maintain issue hygiene, close completed items, and transition issues through defined states. The GitHub REST API's issues.update endpoint provides access to all mutable issue properties.

Common use cases include auto-closing issues when related PRs merge, bulk state changes based on criteria, and maintaining consistent issue lifecycle rules.

=== Usage ===

Use this principle when:
- Automating issue closure after related PRs merge
- Building issue triage or cleanup workflows
- Implementing issue state machines
- Maintaining issue hygiene at scale

== Theoretical Basis ==

=== Issue States ===
<syntaxhighlight lang="text">
Primary States:
- open: Active issue
- closed: Resolved issue

Closed Reason (GitHub API v3):
- completed: Successfully resolved
- not_planned: Won't fix / out of scope
</syntaxhighlight>

=== API Operations ===
<syntaxhighlight lang="javascript">
// Close an issue
await github.rest.issues.update({
  owner: "owner",
  repo: "repo",
  issue_number: 42,
  state: "closed"
});

// Reopen an issue
await github.rest.issues.update({
  owner: "owner",
  repo: "repo",
  issue_number: 42,
  state: "open"
});
</syntaxhighlight>

=== Additional Updateable Fields ===
{| class="wikitable"
|-
! Field !! Type !! Description
|-
| title || string || Issue title
|-
| body || string || Issue description
|-
| labels || array || Label names to set
|-
| assignees || array || Usernames to assign
|-
| milestone || number || Milestone ID
|-
| state_reason || string || "completed" or "not_planned"
|}

=== Automation Patterns ===
<syntaxhighlight lang="text">
1. Issue-to-PR Conversion:
   - Create PR from issue data
   - Close original issue
   - PR body references issue

2. Stale Issue Cleanup:
   - Query old issues
   - Add "stale" label
   - Close after warning period

3. PR Merge Trigger:
   - On PR merge event
   - Find linked issues
   - Close them with "completed" reason
</syntaxhighlight>

=== GitHub Script Example ===
<syntaxhighlight lang="yaml">
- uses: actions/github-script@v7
  with:
    script: |
      await github.rest.issues.update({
        owner: context.repo.owner,
        repo: context.repo.repo,
        issue_number: context.issue.number,
        state: 'closed',
        state_reason: 'completed'
      });
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:0PandaDEV_Awesome_windows_close_issue_action]]
