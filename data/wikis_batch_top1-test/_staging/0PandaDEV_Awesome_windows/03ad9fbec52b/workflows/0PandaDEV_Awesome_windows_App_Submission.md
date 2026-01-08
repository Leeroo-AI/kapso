# Workflow: App_Submission

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Awesome Windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|GitHub Actions|https://docs.github.com/en/actions]]
* [[source::Doc|GitHub Issue Forms|https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests]]
|-
! Domains
| [[domain::CI_CD]], [[domain::Automation]], [[domain::Documentation]], [[domain::Community]]
|-
! Last Updated
| [[last_updated::2026-01-08 18:30 GMT]]
|}

== Overview ==
End-to-end process for submitting new applications to the Awesome Windows list through GitHub Issues, with automated conversion to Pull Requests.

=== Description ===
This workflow enables community members to suggest new Windows applications for inclusion in the curated list. Users submit applications via a structured GitHub Issue form that collects application details (name, URL, category, description, attributes). A maintainer can then trigger automated conversion of the issue into a Pull Request by commenting `/convert`. The automation parses the issue body, extracts application metadata, updates the README.md file, and creates a PR that references the original issue.

=== Usage ===
Execute this workflow when you want to add a new application to the Awesome Windows list. Users should create an issue using the "Add Application" template, providing complete information about the application including its category, description, and whether it is open source or paid.

== Execution Steps ==

=== Step 1: Create Submission Issue ===
[[step::Principle:0PandaDEV_Awesome_windows_Issue_Template_Submission]]

User creates a new GitHub issue using the "Add Application" issue template. The form captures structured data including application name, URL, category (from predefined list), description, and attributes (open source, paid, freemium). Repository URL is optional for open source applications.

'''Required fields:'''
* Application Name
* Application URL
* Category (dropdown with 30+ options)
* Description
* Code of Conduct agreement

'''Optional fields:'''
* Open Source / Paid / Freemium checkboxes
* Repository URL (for open source apps)
* Additional information

=== Step 2: Trigger Conversion ===
[[step::Principle:0PandaDEV_Awesome_windows_Comment_Command_Trigger]]

Maintainer reviews the issue and triggers automated conversion by posting a `/convert` comment. The GitHub Actions workflow is triggered by the issue_comment event and validates that the commenter is the authorized maintainer (0pandadev) and the issue has the "Add" label.

'''Authorization checks:'''
* Comment body equals "/convert"
* Commenter is authorized maintainer
* Issue has "Add" label

=== Step 3: Parse Issue Metadata ===
[[step::Principle:0PandaDEV_Awesome_windows_Issue_Body_Parsing]]

The workflow fetches the issue details via GitHub API and parses the structured form data from the issue body. AWK commands extract each field using section headers as delimiters.

'''Extracted fields:'''
* APP_NAME - Application name
* APP_URL - Application website
* CATEGORY - Target category section
* APP_DESCRIPTION - Brief description
* REPO_URL - Source code repository (if open source)
* Attribute flags (Open Source, Paid)

=== Step 4: Generate List Entry ===
[[step::Principle:0PandaDEV_Awesome_windows_List_Entry_Generation]]

Construct the markdown list entry for the application following the awesome-list format. The entry includes the application link, description, and appropriate badges (open source icon with repo link, paid icon).

'''Entry format:'''
* `* [AppName](url) - Description [![oss][oss]](repo_url) ![paid]`
* Icons are conditionally included based on attributes

=== Step 5: Insert Into README ===
[[step::Principle:0PandaDEV_Awesome_windows_Alphabetical_Insertion]]

Insert the new entry into the appropriate category section of README.md, maintaining alphabetical order. The AWK script locates the target category, finds the correct insertion point based on alphabetical comparison, and inserts the new entry.

'''Insertion logic:'''
* Find category section header
* Compare against existing entries
* Insert at alphabetically correct position
* Preserve file structure

=== Step 6: Create Pull Request ===
[[step::Principle:0PandaDEV_Awesome_windows_PR_Creation]]

Create a Pull Request from the changes using the peter-evans/create-pull-request action. The PR includes a descriptive title, body with application details, and automatically references the original issue for closure upon merge.

'''PR metadata:'''
* Title: "Add {AppName} to {Category} category"
* Body: Application details and issue reference
* Branch: add-{issue_number}
* Auto-closes original issue

=== Step 7: Close Original Issue ===
[[step::Principle:0PandaDEV_Awesome_windows_Issue_State_Management]]

After PR creation, automatically close the original issue. This is handled via GitHub API using the actions/github-script action, updating the issue state to "closed".

== Execution Diagram ==
{{#mermaid:graph TD
    A[User Creates Add Issue] --> B[Maintainer Reviews]
    B --> C[Comment /convert]
    C --> D[Parse Issue Metadata]
    D --> E[Generate List Entry]
    E --> F[Insert Into README]
    F --> G[Create Pull Request]
    G --> H[Close Original Issue]
}}

== Related Pages ==
* [[step::Principle:0PandaDEV_Awesome_windows_Issue_Template_Submission]]
* [[step::Principle:0PandaDEV_Awesome_windows_Comment_Command_Trigger]]
* [[step::Principle:0PandaDEV_Awesome_windows_Issue_Body_Parsing]]
* [[step::Principle:0PandaDEV_Awesome_windows_List_Entry_Generation]]
* [[step::Principle:0PandaDEV_Awesome_windows_Alphabetical_Insertion]]
* [[step::Principle:0PandaDEV_Awesome_windows_PR_Creation]]
* [[step::Principle:0PandaDEV_Awesome_windows_Issue_State_Management]]
