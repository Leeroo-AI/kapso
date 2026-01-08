# Workflow: Contributor_Update_Automation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Awesome Windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|GitHub Actions|https://docs.github.com/en/actions]]
* [[source::Doc|GitHub REST API|https://docs.github.com/en/rest]]
|-
! Domains
| [[domain::CI_CD]], [[domain::Automation]], [[domain::Documentation]]
|-
! Last Updated
| [[last_updated::2026-01-08 18:30 GMT]]
|}

== Overview ==
Automated daily process to fetch repository contributors from GitHub API and update the README.md backers section with their avatars.

=== Description ===
This workflow maintains an up-to-date list of contributors in the README.md file. It runs on a scheduled basis (daily via cron) or can be triggered manually. The process fetches contributor data from the GitHub API, compares it against the current README content, and updates the backers section if any new contributors are detected. Each contributor is displayed with a circular avatar image processed through the weserv.nl image proxy service.

=== Usage ===
Execute this workflow when you need to ensure the contributor list reflects all repository contributors. The workflow runs automatically on a daily schedule, but can also be triggered manually via GitHub Actions workflow_dispatch when immediate updates are needed.

== Execution Steps ==

=== Step 1: Fetch Contributors ===
[[step::Principle:0PandaDEV_Awesome_windows_GitHub_API_Integration]]

Retrieve the list of repository contributors from the GitHub REST API. The API call authenticates using a Personal Access Token (PAT) and returns contributor data including usernames and avatar URLs. The bot user (actions-user) is filtered out to show only human contributors.

'''Key considerations:'''
* Requires GITHUB_PAT environment variable for authentication
* Uses GITHUB_REPOSITORY environment variable to target correct repo
* Filters out automated users (actions-user)

=== Step 2: Detect Changes ===
[[step::Principle:0PandaDEV_Awesome_windows_Content_Change_Detection]]

Compare the fetched contributor list against the current README.md content. The detection mechanism checks whether each contributor's GitHub profile URL exists in the current file. If any contributor URL is missing, the workflow proceeds to update the README.

'''What happens:'''
* Reads current README.md content
* Iterates through each contributor
* Checks for presence of GitHub profile URL
* Returns true if any contributor is missing

=== Step 3: Generate Contributor Block ===
[[step::Principle:0PandaDEV_Awesome_windows_README_Section_Generation]]

Generate the new HTML content for the backers section. Each contributor entry consists of an anchor link to their GitHub profile containing a circular avatar image. The avatar images are processed through the weserv.nl proxy to apply circular masking and caching.

'''Output format:'''
* Linked avatar images for each contributor
* Section header and thank-you message
* Support link (Buy Me A Coffee)

=== Step 4: Update README ===
[[step::Principle:0PandaDEV_Awesome_windows_Regex_Content_Replacement]]

Replace the existing backers section in README.md with the newly generated content. The replacement uses regex pattern matching to locate and replace the section between the "## Backers" header and the reference link definitions at the end of the file.

'''Pattern matching:'''
* Multiline regex finds section start (## Backers)
* Captures everything until reference link definitions
* Replaces entire section with new content

=== Step 5: Commit and Push ===
[[step::Principle:0PandaDEV_Awesome_windows_Git_Commit_Automation]]

If changes were detected and the README was updated, commit the changes and push to the repository. The commit is made using the repository owner's identity and a standardized commit message.

'''Automation flow:'''
* Configure git with maintainer identity
* Stage README.md changes
* Commit with message "Update contributors"
* Push to main branch

== Execution Diagram ==
{{#mermaid:graph TD
    A[Fetch Contributors from GitHub API] --> B[Detect Changes in README]
    B --> C{Changes Detected?}
    C -->|Yes| D[Generate Contributor Block]
    C -->|No| E[Exit - No Changes]
    D --> F[Update README with Regex]
    F --> G[Commit and Push Changes]
}}

== Related Pages ==
* [[step::Principle:0PandaDEV_Awesome_windows_GitHub_API_Integration]]
* [[step::Principle:0PandaDEV_Awesome_windows_Content_Change_Detection]]
* [[step::Principle:0PandaDEV_Awesome_windows_README_Section_Generation]]
* [[step::Principle:0PandaDEV_Awesome_windows_Regex_Content_Replacement]]
* [[step::Principle:0PandaDEV_Awesome_windows_Git_Commit_Automation]]
