# Workflow Index: 0PandaDEV_Awesome_windows

> Comprehensive index of Workflows and their implementation context.
> This index bridges Phase 1 (Anchoring) and Phase 2 (Excavation).
> **Update IMMEDIATELY** after creating or modifying a Workflow page.

---

## Summary

| Workflow | Steps | Principles | Rough APIs |
|----------|-------|------------|------------|
| Contributor_Update_Automation | 5 | 5 | requests.get, re.sub, file.read, file.write |
| App_Submission | 7 | 7 | github-script, create-pull-request, awk |

---

## Workflow: 0PandaDEV_Awesome_windows_Contributor_Update_Automation

**File:** [→](./workflows/0PandaDEV_Awesome_windows_Contributor_Update_Automation.md)
**Description:** Automated daily process to fetch repository contributors from GitHub API and update the README.md backers section.

### Steps Overview

| # | Step Name | Principle | Rough API | Related Files |
|---|-----------|-----------|-----------|---------------|
| 1 | Fetch Contributors | GitHub_API_Integration | `requests.get()` | update_contributors.py |
| 2 | Detect Changes | Content_Change_Detection | `file.read()`, string search | update_contributors.py |
| 3 | Generate Contributor Block | README_Section_Generation | String concatenation | update_contributors.py |
| 4 | Update README | Regex_Content_Replacement | `re.sub()` | update_contributors.py |
| 5 | Commit and Push | Git_Commit_Automation | `git add`, `git commit`, `git push` | update_contributors.yml |

### Source Files (for enrichment)

- `.github/scripts/update_contributors.py` - Main Python script with API calls and README manipulation
- `.github/workflows/update_contributors.yml` - GitHub Actions workflow definition

### Step 1: GitHub_API_Integration

| Attribute | Value |
|-----------|-------|
| **Principle** | `0PandaDEV_Awesome_windows_GitHub_API_Integration` |
| **Implementation** | `0PandaDEV_Awesome_windows_get_contributors` |
| **API Call** | `get_contributors() -> list[dict]` |
| **Source Location** | `.github/scripts/update_contributors.py:L6-11` |
| **External Dependencies** | `requests`, `os` |
| **Environment** | `0PandaDEV_Awesome_windows_GitHub_Actions_Python` |
| **Key Parameters** | `GITHUB_PAT: str` - Personal access token from environment, `GITHUB_REPOSITORY: str` - Repository name from environment |
| **Inputs** | GitHub API endpoint `https://api.github.com/repos/{repo}/contributors`, PAT token |
| **Outputs** | List of contributor dictionaries (excluding 'actions-user'), each containing `login`, `avatar_url` |

### Step 2: Content_Change_Detection

| Attribute | Value |
|-----------|-------|
| **Principle** | `0PandaDEV_Awesome_windows_Content_Change_Detection` |
| **Implementation** | `0PandaDEV_Awesome_windows_has_contributors_changed` |
| **API Call** | `has_contributors_changed(contributors: list[dict]) -> bool` |
| **Source Location** | `.github/scripts/update_contributors.py:L14-22` |
| **External Dependencies** | (none - built-in file I/O) |
| **Environment** | `0PandaDEV_Awesome_windows_GitHub_Actions_Python` |
| **Key Parameters** | `contributors: list[dict]` - List of contributor dicts with `login` key |
| **Inputs** | Contributors list from Step 1, `README.md` file content |
| **Outputs** | Boolean indicating whether any contributor GitHub URL is missing from README |

### Step 3: README_Section_Generation

| Attribute | Value |
|-----------|-------|
| **Principle** | `0PandaDEV_Awesome_windows_README_Section_Generation` |
| **Implementation** | `0PandaDEV_Awesome_windows_update_readme_generation` |
| **API Call** | `update_readme(contributors: list[dict]) -> None` (generation portion) |
| **Source Location** | `.github/scripts/update_contributors.py:L25-36` |
| **External Dependencies** | (none - string operations) |
| **Environment** | `0PandaDEV_Awesome_windows_GitHub_Actions_Python` |
| **Key Parameters** | `contributors: list[dict]` - List with `login` and `avatar_url` keys |
| **Inputs** | Contributors list from Step 1 |
| **Outputs** | Markdown string block containing "## Backers" header, contributor avatar links (via weserv.nl proxy), and support message |

### Step 4: Regex_Content_Replacement

| Attribute | Value |
|-----------|-------|
| **Principle** | `0PandaDEV_Awesome_windows_Regex_Content_Replacement` |
| **Implementation** | `0PandaDEV_Awesome_windows_update_readme_replacement` |
| **API Call** | `re.sub(pattern: str, repl: str, string: str) -> str` |
| **Source Location** | `.github/scripts/update_contributors.py:L38-41` |
| **External Dependencies** | `re` |
| **Environment** | `0PandaDEV_Awesome_windows_GitHub_Actions_Python` |
| **Key Parameters** | `pattern: str` - `r"(?ms)^## Backers\s*\n.*?(?=^\[oss\]:)"` multiline regex |
| **Inputs** | README.md content, new Backers block from Step 3 |
| **Outputs** | Updated README.md content with replaced Backers section, written to file |

### Step 5: Git_Commit_Automation

| Attribute | Value |
|-----------|-------|
| **Principle** | `0PandaDEV_Awesome_windows_Git_Commit_Automation` |
| **Implementation** | `0PandaDEV_Awesome_windows_git_commit_push` |
| **API Call** | `git add README.md && git commit -m "Update contributors" && git push` |
| **Source Location** | `.github/workflows/update_contributors.yml:L34-40` |
| **External Dependencies** | `git` CLI |
| **Environment** | `0PandaDEV_Awesome_windows_GitHub_Actions_Ubuntu` |
| **Key Parameters** | `user.email: str` - Git author email, `user.name: str` - Git author name |
| **Inputs** | Modified README.md from Step 4, conditional on `update_status == 'Contributors updated'` |
| **Outputs** | Git commit pushed to repository main branch |

### Implementation Extraction Guide

| Principle | Implementation | API | Source | Type |
|-----------|----------------|-----|--------|------|
| GitHub_API_Integration | `get_contributors` | `requests.get()` | update_contributors.py:L6-11 | API Doc |
| Content_Change_Detection | `has_contributors_changed` | `file.read()`, `in` operator | update_contributors.py:L14-22 | API Doc |
| README_Section_Generation | `update_readme_generation` | String concatenation | update_contributors.py:L25-36 | API Doc |
| Regex_Content_Replacement | `update_readme_replacement` | `re.sub()` | update_contributors.py:L38-41 | API Doc |
| Git_Commit_Automation | `git_commit_push` | `git` CLI | update_contributors.yml:L34-40 | External Tool Doc |

---

## Workflow: 0PandaDEV_Awesome_windows_App_Submission

**File:** [→](./workflows/0PandaDEV_Awesome_windows_App_Submission.md)
**Description:** End-to-end process for submitting new applications via GitHub Issues with automated PR conversion.

### Steps Overview

| # | Step Name | Principle | Rough API | Related Files |
|---|-----------|-----------|-----------|---------------|
| 1 | Create Submission Issue | Issue_Template_Submission | GitHub Issue Form | add_app.yml |
| 2 | Trigger Conversion | Comment_Command_Trigger | `issue_comment` event | covert_to_pr.yml |
| 3 | Parse Issue Metadata | Issue_Body_Parsing | `github-script`, `awk` | covert_to_pr.yml |
| 4 | Generate List Entry | List_Entry_Generation | Shell string manipulation | covert_to_pr.yml |
| 5 | Insert Into README | Alphabetical_Insertion | `awk` script | covert_to_pr.yml |
| 6 | Create Pull Request | PR_Creation | `peter-evans/create-pull-request` | covert_to_pr.yml |
| 7 | Close Original Issue | Issue_State_Management | `github-script` | covert_to_pr.yml |

### Source Files (for enrichment)

- `.github/ISSUE_TEMPLATE/add_app.yml` - Issue form template defining submission fields
- `.github/workflows/covert_to_pr.yml` - GitHub Actions workflow for issue-to-PR conversion

### Step 1: Issue_Template_Submission

| Attribute | Value |
|-----------|-------|
| **Principle** | `0PandaDEV_Awesome_windows_Issue_Template_Submission` |
| **Implementation** | `0PandaDEV_Awesome_windows_add_app_form` |
| **API Call** | GitHub Issue Forms YAML schema |
| **Source Location** | `.github/ISSUE_TEMPLATE/add_app.yml:L1-117` |
| **External Dependencies** | GitHub Issue Forms |
| **Environment** | `0PandaDEV_Awesome_windows_GitHub_Issues` |
| **Key Parameters** | `app-name: input` - Application name, `app-url: input` - Application URL, `app-category: dropdown` - Category selection, `app-description: textarea` - Description, `app-attributes: checkboxes` - Open Source/Paid flags, `repo-url: input` - Repository URL (optional) |
| **Inputs** | User-submitted form data via GitHub web interface |
| **Outputs** | GitHub Issue with structured body containing all form fields, labeled with "Add" |

### Step 2: Comment_Command_Trigger

| Attribute | Value |
|-----------|-------|
| **Principle** | `0PandaDEV_Awesome_windows_Comment_Command_Trigger` |
| **Implementation** | `0PandaDEV_Awesome_windows_convert_command_check` |
| **API Call** | GitHub Actions `if` conditional expression |
| **Source Location** | `.github/workflows/covert_to_pr.yml:L15-20` |
| **External Dependencies** | GitHub Actions |
| **Environment** | `0PandaDEV_Awesome_windows_GitHub_Actions_Ubuntu` |
| **Key Parameters** | `comment.body: str` - Must equal `/convert`, `comment.user.login: str` - Must be `0pandadev`, `issue.labels: array` - Must contain "Add" |
| **Inputs** | `issue_comment` event or `workflow_dispatch` with `issue_number` input |
| **Outputs** | Job execution triggered if conditions met |

### Step 3: Issue_Body_Parsing

| Attribute | Value |
|-----------|-------|
| **Principle** | `0PandaDEV_Awesome_windows_Issue_Body_Parsing` |
| **Implementation** | `0PandaDEV_Awesome_windows_issue_metadata_extraction` |
| **API Call** | `github.rest.issues.get()` + `awk` field extraction |
| **Source Location** | `.github/workflows/covert_to_pr.yml:L28-51` |
| **External Dependencies** | `actions/github-script@v7`, `awk` |
| **Environment** | `0PandaDEV_Awesome_windows_GitHub_Actions_Ubuntu` |
| **Key Parameters** | `issueNumber: number` - Issue to fetch, `awk pattern: str` - `/### Field Name/{flag=1; next} /###/{flag=0} flag` |
| **Inputs** | Issue body from `github.rest.issues.get()` |
| **Outputs** | Shell variables: `APP_NAME`, `APP_URL`, `CATEGORY`, `APP_DESCRIPTION`, `REPO_URL` |

### Step 4: List_Entry_Generation

| Attribute | Value |
|-----------|-------|
| **Principle** | `0PandaDEV_Awesome_windows_List_Entry_Generation` |
| **Implementation** | `0PandaDEV_Awesome_windows_entry_builder` |
| **API Call** | Shell string concatenation and conditionals |
| **Source Location** | `.github/workflows/covert_to_pr.yml:L53-72` |
| **External Dependencies** | `bash`, `grep` |
| **Environment** | `0PandaDEV_Awesome_windows_GitHub_Actions_Ubuntu` |
| **Key Parameters** | `OPEN_SOURCE_ICON: str` - `[![Open-Source Software][oss]]($REPO_URL)` if checked, `PAID_ICON: str` - `![paid]` if checked |
| **Inputs** | Parsed variables from Step 3, checkbox states from issue body |
| **Outputs** | `NEW_ENTRY` string in format: `* [$APP_NAME]($APP_URL) - $APP_DESCRIPTION $OPEN_SOURCE_ICON $PAID_ICON` |

### Step 5: Alphabetical_Insertion

| Attribute | Value |
|-----------|-------|
| **Principle** | `0PandaDEV_Awesome_windows_Alphabetical_Insertion` |
| **Implementation** | `0PandaDEV_Awesome_windows_awk_insert_sorted` |
| **API Call** | `awk` script with category matching and alphabetical comparison |
| **Source Location** | `.github/workflows/covert_to_pr.yml:L75-99` |
| **External Dependencies** | `awk` |
| **Environment** | `0PandaDEV_Awesome_windows_GitHub_Actions_Ubuntu` |
| **Key Parameters** | `new_entry: str` - Entry to insert, `category: str` - Target section header |
| **Inputs** | `NEW_ENTRY` from Step 4, `CATEGORY` from Step 3, `README.md` content |
| **Outputs** | Modified `README.md` with new entry inserted alphabetically within the correct category section |

### Step 6: PR_Creation

| Attribute | Value |
|-----------|-------|
| **Principle** | `0PandaDEV_Awesome_windows_PR_Creation` |
| **Implementation** | `0PandaDEV_Awesome_windows_create_pull_request_action` |
| **API Call** | `peter-evans/create-pull-request@v6` |
| **Source Location** | `.github/workflows/covert_to_pr.yml:L120-134` |
| **External Dependencies** | `peter-evans/create-pull-request@v6` |
| **Environment** | `0PandaDEV_Awesome_windows_GitHub_Actions_Ubuntu` |
| **Key Parameters** | `token: str` - PAT secret, `commit-message: str` - Auto-generated, `title: str` - "Add {APP_NAME} to {CATEGORY} category", `branch: str` - `add-{issue_number}`, `base: str` - `main` |
| **Inputs** | Modified README.md from Step 5, environment variables from Step 3 |
| **Outputs** | Pull request created with auto-generated title, body including app details, and `Closes #issue_number` reference |

### Step 7: Issue_State_Management

| Attribute | Value |
|-----------|-------|
| **Principle** | `0PandaDEV_Awesome_windows_Issue_State_Management` |
| **Implementation** | `0PandaDEV_Awesome_windows_close_issue_action` |
| **API Call** | `github.rest.issues.update({ state: 'closed' })` |
| **Source Location** | `.github/workflows/covert_to_pr.yml:L136-147` |
| **External Dependencies** | `actions/github-script@v7` |
| **Environment** | `0PandaDEV_Awesome_windows_GitHub_Actions_Ubuntu` |
| **Key Parameters** | `owner: str` - Repository owner, `repo: str` - Repository name, `issue_number: number` - Issue to close, `state: str` - `'closed'` |
| **Inputs** | Issue number from trigger context |
| **Outputs** | Original submission issue closed |

### Implementation Extraction Guide

| Principle | Implementation | API | Source | Type |
|-----------|----------------|-----|--------|------|
| Issue_Template_Submission | `add_app_form` | GitHub Issue Forms YAML | add_app.yml:L1-117 | Pattern Doc |
| Comment_Command_Trigger | `convert_command_check` | GitHub Actions `if` | covert_to_pr.yml:L15-20 | Pattern Doc |
| Issue_Body_Parsing | `issue_metadata_extraction` | `github-script`, `awk` | covert_to_pr.yml:L28-51 | External Tool Doc |
| List_Entry_Generation | `entry_builder` | Shell variables | covert_to_pr.yml:L53-72 | Pattern Doc |
| Alphabetical_Insertion | `awk_insert_sorted` | `awk` script | covert_to_pr.yml:L75-99 | External Tool Doc |
| PR_Creation | `create_pull_request_action` | `peter-evans/create-pull-request` | covert_to_pr.yml:L120-134 | Wrapper Doc |
| Issue_State_Management | `close_issue_action` | `github.rest.issues.update` | covert_to_pr.yml:L136-147 | Wrapper Doc |

---

## Global Implementation Extraction Guide

| Principle | Implementation | API | Source | Type |
|-----------|----------------|-----|--------|------|
| GitHub_API_Integration | `get_contributors` | `requests.get()` | update_contributors.py:L6-11 | API Doc |
| Content_Change_Detection | `has_contributors_changed` | `file.read()`, `in` operator | update_contributors.py:L14-22 | API Doc |
| README_Section_Generation | `update_readme_generation` | String concatenation | update_contributors.py:L25-36 | API Doc |
| Regex_Content_Replacement | `update_readme_replacement` | `re.sub()` | update_contributors.py:L38-41 | API Doc |
| Git_Commit_Automation | `git_commit_push` | `git` CLI | update_contributors.yml:L34-40 | External Tool Doc |
| Issue_Template_Submission | `add_app_form` | GitHub Issue Forms YAML | add_app.yml:L1-117 | Pattern Doc |
| Comment_Command_Trigger | `convert_command_check` | GitHub Actions `if` | covert_to_pr.yml:L15-20 | Pattern Doc |
| Issue_Body_Parsing | `issue_metadata_extraction` | `github-script`, `awk` | covert_to_pr.yml:L28-51 | External Tool Doc |
| List_Entry_Generation | `entry_builder` | Shell variables | covert_to_pr.yml:L53-72 | Pattern Doc |
| Alphabetical_Insertion | `awk_insert_sorted` | `awk` script | covert_to_pr.yml:L75-99 | External Tool Doc |
| PR_Creation | `create_pull_request_action` | `peter-evans/create-pull-request` | covert_to_pr.yml:L120-134 | Wrapper Doc |
| Issue_State_Management | `close_issue_action` | `github.rest.issues.update` | covert_to_pr.yml:L136-147 | Wrapper Doc |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation

**Implementation Types:**
- **API Doc:** Function/class in this repo
- **Wrapper Doc:** External library with repo-specific usage
- **Pattern Doc:** User-defined interface/pattern
- **External Tool Doc:** CLI or external tool
