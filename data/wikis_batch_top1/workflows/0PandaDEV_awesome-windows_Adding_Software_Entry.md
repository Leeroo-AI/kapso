{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|0PandaDEV_awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|Awesome List Guidelines|https://github.com/sindresorhus/awesome/blob/main/contributing.md]]
|-
! Domains
| [[domain::Open_Source]], [[domain::Curation]], [[domain::Documentation]]
|-
! Last Updated
| [[last_updated::2026-01-08 14:00 GMT]]
|}

== Overview ==

End-to-end process for contributing a new Windows software entry to the awesome-windows curated list via GitHub Issue or manual Pull Request.

=== Description ===

This workflow outlines the complete contribution process for adding a new Windows application to the curated list. Contributors can choose between two paths: (1) an automated path using GitHub Issue templates that auto-generates pull requests, or (2) a manual path by forking and directly editing the README. Both paths result in a properly formatted entry with application name, URL, description, and appropriate icons (open-source, paid) inserted in alphabetical order within the correct category section.

=== Usage ===

Execute this workflow when you want to suggest a Windows application for inclusion in the awesome-windows list. Prerequisites: a GitHub account and knowledge of the application's URL, category, description, and whether it is open-source or paid software.

== Execution Steps ==

=== Step 1: Prepare Application Information ===
[[step::Principle:0PandaDEV_awesome-windows_Application_Information_Gathering]]

Collect all required information about the application before submission. This includes the application name, official website URL, description of functionality, appropriate category from the defined list, and repository URL if open-source.

'''Key considerations:'''
* Verify the application actually runs on Windows
* Check the list to ensure the application is not already included
* Determine if the software is open-source, paid, or freemium
* Write a concise description focusing on what the application does

=== Step 2: Choose Submission Path ===
[[step::Principle:0PandaDEV_awesome-windows_Submission_Path_Selection]]

Select between the automated GitHub Issue approach or the manual Pull Request approach based on familiarity with Git and preference.

'''Paths available:'''
* Issue Template Path: For users who prefer a form-based submission that auto-generates PRs
* Manual PR Path: For users comfortable with forking, editing markdown, and creating pull requests directly

=== Step 3: Submit via Issue Template (Automated Path) ===
[[step::Principle:0PandaDEV_awesome-windows_Issue_Template_Submission]]

Navigate to the GitHub repository's Issues tab and select the "Add Application" template. Fill out all required fields including application name, URL, category, and description. Check applicable attributes (Open Source, Paid, Freemium) and provide repository URL if open-source.

'''What happens after submission:'''
* Issue is labeled with "Add" tag
* Maintainer reviews and comments "/convert" to trigger automation
* GitHub Actions workflow parses issue fields and generates a PR automatically

=== Step 4: Submit via Manual Pull Request (Manual Path) ===
[[step::Principle:0PandaDEV_awesome-windows_Manual_PR_Submission]]

Fork the repository, locate the correct category section in README.md, and insert the new entry in alphabetical order following the established format. Update the Contents section if adding a new category.

'''Entry format:'''
* Standard: `* [App Name](URL) - Description.`
* With OSS icon: `* [App Name](URL) - Description. [![Open-Source Software][oss]](repo-url)`
* With paid icon: `* [App Name](URL) - Description. ![paid]`

=== Step 5: Pass Automated Linting ===
[[step::Principle:0PandaDEV_awesome-windows_Awesome_Lint_Validation]]

The pull request triggers the awesome-lint GitHub Action which validates the README against sindresorhus/awesome guidelines. Ensure proper formatting, valid URLs, alphabetical ordering, and correct markdown syntax.

'''Common validation checks:'''
* Links are valid and accessible
* Entries follow the prescribed format
* No duplicate entries exist
* Alphabetical ordering is maintained within categories

=== Step 6: Maintainer Review and Merge ===
[[step::Principle:0PandaDEV_awesome-windows_PR_Review_Process]]

The repository maintainer reviews the pull request for quality, relevance, and guideline adherence. Feedback may be provided requiring amendments to spelling, formatting, or description clarity before approval.

'''Review criteria:'''
* Application is useful and appropriate for the list
* Description accurately reflects the application's purpose
* Formatting matches existing entries
* No spelling or grammar errors

== Execution Diagram ==

{{#mermaid:graph TD
    A[Prepare Application Information] --> B{Choose Submission Path}
    B -->|Automated| C[Submit via Issue Template]
    B -->|Manual| D[Submit via Manual PR]
    C --> E[Maintainer Converts Issue to PR]
    E --> F[Pass Automated Linting]
    D --> F
    F --> G[Maintainer Review and Merge]
}}

== Related Pages ==

=== Steps ===
* [[step::Principle:0PandaDEV_awesome-windows_Application_Information_Gathering]]
* [[step::Principle:0PandaDEV_awesome-windows_Submission_Path_Selection]]
* [[step::Principle:0PandaDEV_awesome-windows_Issue_Template_Submission]]
* [[step::Principle:0PandaDEV_awesome-windows_Manual_PR_Submission]]
* [[step::Principle:0PandaDEV_awesome-windows_Awesome_Lint_Validation]]
* [[step::Principle:0PandaDEV_awesome-windows_PR_Review_Process]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:0PandaDEV_awesome-windows_Alphabetical_Ordering_Convention]]
* [[uses_heuristic::Heuristic:0PandaDEV_awesome-windows_AP_Style_Title_Casing]]
