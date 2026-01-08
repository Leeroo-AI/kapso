# Workflow Index: 0PandaDEV_awesome-windows

> Comprehensive index of Workflows and their implementation context.
> This index bridges Phase 1 (Anchoring) and Phase 2 (Excavation).
> **Update IMMEDIATELY** after creating or modifying a Workflow page.

---

## Summary

| Workflow | Steps | Principles | Rough APIs |
|----------|-------|------------|------------|
| Adding_Software_Entry | 6 | 6 | Issue template parsing, README editing, awesome-lint |
| Automated_Contributor_Update | 6 | 6 | requests.get, GitHub API, regex, weserv.nl |

---

## Workflow: 0PandaDEV_awesome-windows_Adding_Software_Entry

**File:** [→](./workflows/0PandaDEV_awesome-windows_Adding_Software_Entry.md)
**Description:** End-to-end process for contributing a new Windows software entry to the awesome-windows curated list.

### Steps Overview

| # | Step Name | Principle | Rough API | Related Files |
|---|-----------|-----------|-----------|---------------|
| 1 | Prepare Application Information | Application_Information_Gathering | N/A (manual) | README.md |
| 2 | Choose Submission Path | Submission_Path_Selection | N/A (decision point) | CONTRIBUTING.md |
| 3 | Submit via Issue Template | Issue_Template_Submission | GitHub Issue Forms API | .github/ISSUE_TEMPLATE/add_app.yml |
| 4 | Submit via Manual PR | Manual_PR_Submission | Git fork/edit workflow | README.md |
| 5 | Pass Automated Linting | Awesome_Lint_Validation | awesome-lint-action | .github/workflows/awesome-lint.yml |
| 6 | Maintainer Review and Merge | PR_Review_Process | peter-evans/create-pull-request | .github/workflows/covert_to_pr.yml |

### Source Files (for enrichment)

- `CONTRIBUTING.md` - Contribution guidelines with formatting requirements
- `.github/ISSUE_TEMPLATE/add_app.yml` - Issue form schema for new applications
- `.github/workflows/covert_to_pr.yml` - Automation that converts issues to PRs
- `.github/workflows/awesome-lint.yml` - Linting workflow for PR validation

<!-- ENRICHMENT NEEDED: Phase 1b will add detailed Step N attribute tables below -->

---

## Workflow: 0PandaDEV_awesome-windows_Automated_Contributor_Update

**File:** [→](./workflows/0PandaDEV_awesome-windows_Automated_Contributor_Update.md)
**Description:** Automated CI/CD workflow that updates the README Backers section with contributor avatars from GitHub API.

### Steps Overview

| # | Step Name | Principle | Rough API | Related Files |
|---|-----------|-----------|-----------|---------------|
| 1 | Trigger Workflow | Workflow_Trigger_Scheduling | GitHub Actions cron | .github/workflows/update_contributors.yml |
| 2 | Fetch Contributors from GitHub API | GitHub_API_Integration | `requests.get()` GitHub Contributors API | .github/scripts/update_contributors.py:L6-11 |
| 3 | Check for Changes | Change_Detection | String search in README | .github/scripts/update_contributors.py:L14-22 |
| 4 | Generate Contributor HTML | Avatar_HTML_Generation | weserv.nl URL generation | .github/scripts/update_contributors.py:L25-36 |
| 5 | Update README Content | README_Content_Update | `re.sub()` regex replacement | .github/scripts/update_contributors.py:L38-41 |
| 6 | Commit and Push Changes | Git_Commit_Automation | git add/commit/push | .github/workflows/update_contributors.yml:L34-40 |

### Source Files (for enrichment)

- `.github/scripts/update_contributors.py` - Main Python script (50 lines)
- `.github/workflows/update_contributors.yml` - GitHub Actions workflow definition

<!-- ENRICHMENT NEEDED: Phase 1b will add detailed Step N attribute tables below -->

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation

**Implementation Types:**
- **API Doc:** Function/class in this repo
- **Wrapper Doc:** External library with repo-specific usage
- **Pattern Doc:** User-defined interface/pattern
- **External Tool Doc:** CLI or external tool
