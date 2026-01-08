# Workflow: Editing Software Entry

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|Awesome List Guidelines|https://github.com/sindresorhus/awesome/blob/main/contributing.md]]
|-
! Domains
| [[domain::Documentation]], [[domain::Community_Contribution]]
|-
! Last Updated
| [[last_updated::2026-01-08 14:00 GMT]]
|}

== Overview ==

End-to-end process for suggesting edits to existing Windows software entries in the awesome-windows curated list.

=== Description ===

This workflow enables community members to propose modifications to existing entries in the awesome-windows list. Unlike adding new software, editing focuses on updating stale information, correcting errors, changing categories, or removing discontinued applications.

The workflow uses GitHub Issue Forms to collect structured edit requests and routes them to maintainers for review. This ensures quality control while allowing the community to help maintain list accuracy.

=== Usage ===

Execute this workflow when:
- An application's URL has changed
- The application's description is outdated or inaccurate
- The application should be recategorized
- The application's licensing model has changed (e.g., became paid/open source)
- An application should be removed (discontinued, abandoned, or no longer meets quality standards)

== Execution Steps ==

=== Step 1: Identify Edit Requirement ===
[[step::Principle:0PandaDEV_awesome-windows_Edit_Requirement_Identification]]

Identify which existing entry needs modification and what type of change is required. Review the current entry in the README to understand its current state.

'''Key considerations:'''
* Verify the application actually exists in the list
* Identify the specific fields that need updating (URL, description, category, attributes)
* Determine if the edit is a correction, update, or removal

=== Step 2: Submit Edit Request ===
[[step::Principle:0PandaDEV_awesome-windows_Edit_Request_Submission]]

Use the "Edit Application" issue template to submit a structured edit request. The form collects all relevant information needed for maintainers to evaluate and implement the change.

'''Form fields:'''
* Application name and current URL (required)
* Type of edit (URL update, description, category, attributes, removal)
* Detailed description of proposed changes
* New values for changed fields
* Supporting information (reason for change, sources)

=== Step 3: Maintainer Review ===
[[step::Principle:0PandaDEV_awesome-windows_Edit_Review_Process]]

A maintainer reviews the edit request, validates the proposed changes, and either implements them directly or requests additional information.

'''Review criteria:'''
* Accuracy of proposed changes
* Compliance with awesome-list guidelines
* Quality and relevance of updated information

== Execution Diagram ==

{{#mermaid:graph TD
    A[Identify Edit Requirement] --> B[Submit Edit Request]
    B --> C[Maintainer Review]
}}

== Related Pages ==

=== Steps ===
* [[step::Principle:0PandaDEV_awesome-windows_Edit_Requirement_Identification]]
* [[step::Principle:0PandaDEV_awesome-windows_Edit_Request_Submission]]
* [[step::Principle:0PandaDEV_awesome-windows_Edit_Review_Process]]

=== Heuristics ===
* [[uses_heuristic::Heuristic:0PandaDEV_awesome-windows_Alphabetical_Ordering_Convention]]
* [[uses_heuristic::Heuristic:0PandaDEV_awesome-windows_AP_Style_Title_Casing]]
