# Principle: Submission Path Selection

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|GitHub Contributing Guide|https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions]]
|-
! Domains
| [[domain::Documentation]], [[domain::Community_Contribution]], [[domain::Decision_Making]]
|-
! Last Updated
| [[last_updated::2026-01-08 13:00 GMT]]
|}

== Overview ==

Principle for choosing between automated Issue Template submission versus manual Pull Request contribution paths.

=== Description ===

Submission Path Selection is a decision point in the contribution workflow where a contributor must choose how to submit their application entry. The awesome-windows repository supports two paths:

1. '''Issue Template Path''' - Structured form submission that triggers automated PR creation
2. '''Manual PR Path''' - Traditional fork-edit-commit-PR workflow with full control

This principle guides contributors to select the appropriate path based on their comfort level with Git, need for customization, and preference for automation.

=== Usage ===

Apply this principle after gathering application information (Step 1) and before beginning the actual submission. Consider:

- '''Choose Issue Template''' when:
  - You want the fastest, simplest submission process
  - You're less familiar with Git/GitHub workflows
  - Your entry fits standard formatting requirements

- '''Choose Manual PR''' when:
  - You need custom formatting or special cases
  - You're comfortable with fork-based workflows
  - You want to make multiple changes in one PR

== Theoretical Basis ==

'''Decision Matrix:'''

{| class="wikitable"
|-
! Factor !! Issue Template !! Manual PR
|-
| Git Knowledge Required || Low || High
|-
| Automation Level || Full (auto-PR creation) || None (manual steps)
|-
| Customization || Limited to form fields || Full control
|-
| Time to Submit || ~2 minutes || ~5-10 minutes
|-
| Error Handling || Guided validation || Self-verified
|}

'''Trade-offs:'''
* Issue Template prioritizes accessibility over flexibility
* Manual PR prioritizes control over convenience
* Both paths eventually trigger the same linting workflow

== Practical Guide ==

=== Decision Flowchart ===

<syntaxhighlight lang="text">
START
  │
  ├─ Comfortable with Git/GitHub? ─────────────┐
  │                                             │
  NO                                           YES
  │                                             │
  ▼                                             ▼
Issue Template Path                    Need special formatting?
  │                                             │
  │                                    NO       │       YES
  │                                     │       │        │
  │                                     ▼       │        ▼
  │                              Issue Template │   Manual PR Path
  │                                    Path     │
  │                                             │
  ▼                                             ▼
(Continue to Step 3)                   (Continue to Step 4)
</syntaxhighlight>

=== Path Comparison ===

'''Issue Template Path (Step 3):'''
1. Navigate to Issues → New Issue
2. Select "Add Application" template
3. Fill in form fields
4. Submit issue
5. Wait for maintainer to run /convert command
6. Automated PR is created

'''Manual PR Path (Step 4):'''
1. Fork the repository
2. Edit README.md directly
3. Follow alphabetical ordering
4. Use correct markdown format
5. Commit changes
6. Open Pull Request

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:0PandaDEV_awesome-windows_Contribution_Method_Decision]]

=== Part of Workflow ===
* [[workflow::Workflow:0PandaDEV_awesome-windows_Adding_Software_Entry]]
