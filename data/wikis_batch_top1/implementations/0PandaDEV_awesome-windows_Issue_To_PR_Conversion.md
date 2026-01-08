# Implementation: Issue To PR Conversion

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Repo|create-pull-request|https://github.com/peter-evans/create-pull-request]]
* [[source::Repo|github-script|https://github.com/actions/github-script]]
|-
! Domains
| [[domain::CI_CD]], [[domain::GitHub_Actions]], [[domain::Automation]]
|-
! Last Updated
| [[last_updated::2026-01-08 13:00 GMT]]
|}

== Overview ==

External tool documentation for the GitHub Actions workflow that converts issue template submissions into pull requests.

=== Description ===

This is an '''External Tool Doc''' describing the `covert_to_pr.yml` workflow that automates PR creation from issue submissions. It uses AWK scripting to parse issue content, generate README entries, and `peter-evans/create-pull-request` to create the PR.

=== Usage ===

Triggered by maintainer commenting `/convert` on an issue with the "Add" label, or via manual `workflow_dispatch` with an issue number input.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/0PandaDEV/awesome-windows awesome-windows]
* '''File:''' .github/workflows/covert_to_pr.yml:L1-148

=== External References ===
* [https://github.com/peter-evans/create-pull-request peter-evans/create-pull-request v6]
* [https://github.com/actions/github-script actions/github-script v7]
* [https://github.com/actions/checkout actions/checkout v4]

=== Workflow Definition ===

<syntaxhighlight lang="yaml">
name: Convert Issue to Pull Request

on:
  issue_comment:
    types: [created]
  workflow_dispatch:
    inputs:
      issue_number:
        description: 'Issue number to convert to PR'
        required: true
        type: number

jobs:
  convert_issue_to_pr:
    if: |
      (github.event_name == 'issue_comment' &&
      github.event.comment.body == '/convert' &&
      github.event.comment.user.login == '0pandadev' &&
      contains(github.event.issue.labels.*.name, 'Add')) ||
      github.event_name == 'workflow_dispatch'
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          token: ${{ secrets.PAT }}

      - name: Get issue details
        id: issue
        uses: actions/github-script@v7
        with:
          github-token: ${{ secrets.PAT }}
          script: |
            const issueNumber = context.payload.inputs ?
              context.payload.inputs.issue_number : context.issue.number;
            const issue = await github.rest.issues.get({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: issueNumber
            });
            return issue.data;

      - name: Update README.md
        run: |
          # Extract fields from issue body using AWK
          APP_NAME=$(echo "$ISSUE_BODY" | awk '/### Application Name/{flag=1; next} /###/{flag=0} flag' | xargs)
          APP_URL=$(echo "$ISSUE_BODY" | awk '/### Application URL/{flag=1; next} /###/{flag=0} flag' | xargs)
          CATEGORY=$(echo "$ISSUE_BODY" | awk '/### Category/{flag=1; next} /###/{flag=0} flag' | xargs)
          APP_DESCRIPTION=$(echo "$ISSUE_BODY" | awk '/### Description/{flag=1; next} /###/{flag=0} flag' | xargs)
          REPO_URL=$(echo "$ISSUE_BODY" | awk '/### Repository URL/{flag=1; next} /###/{flag=0} flag' | xargs)

          # Determine icons
          if echo "$ISSUE_BODY" | grep -q "\[X\] Open Source"; then
            OPEN_SOURCE_ICON="[![Open-Source Software][oss]]($REPO_URL)"
          fi
          if echo "$ISSUE_BODY" | grep -q "\[X\] Paid"; then
            PAID_ICON="![paid]"
          fi

          # Create entry and insert alphabetically
          NEW_ENTRY="* [$APP_NAME]($APP_URL) - $APP_DESCRIPTION $OPEN_SOURCE_ICON $PAID_ICON"

          # AWK script inserts entry in alphabetical order
          awk -v new_entry="$NEW_ENTRY" -v category="$CATEGORY" '...' README.md

      - name: Create Pull Request
        uses: peter-evans/create-pull-request@v6
        with:
          token: ${{ secrets.PAT }}
          commit-message: "Add ${{ env.APP_NAME }} to ${{ env.CATEGORY }} category"
          title: "Add ${{ env.APP_NAME }} to ${{ env.CATEGORY }} category"
          body: |
            This PR adds ${{ env.APP_NAME }} to the ${{ env.CATEGORY }} category.

            Application URL: ${{ env.APP_URL }}
            Repository URL: ${{ env.REPO_URL }}

            Closes #${{ github.event.issue.number || github.event.inputs.issue_number }}
          branch: add-${{ github.event.issue.number || github.event.inputs.issue_number }}
          base: main

      - name: Close Issue
        uses: actions/github-script@v7
        with:
          github-token: ${{ secrets.PAT }}
          script: |
            const issueNumber = context.payload.inputs ?
              context.payload.inputs.issue_number : context.issue.number;
            await github.rest.issues.update({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: issueNumber,
              state: 'closed'
            });
</syntaxhighlight>

=== AWK Alphabetical Insertion Script ===

<syntaxhighlight lang="awk">
# Inserts new_entry in alphabetical order within category section
BEGIN {in_category=0; added=0}
/^## / {
  if (in_category && !added) {
    print new_entry
    added=1
  }
  in_category = ($0 ~ "^## " category)
  print
  if (in_category) print ""
  next
}
in_category && /^\* / {
  if (!added && tolower(substr(new_entry, 3)) < tolower(substr($0, 3))) {
    print new_entry
    added=1
  }
  print
  next
}
{print}
END {
  if (in_category && !added) print new_entry
}
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| /convert comment || String || Yes* || Trigger from maintainer on issue
|-
| issue_number || Number || Yes* || For workflow_dispatch, the issue to convert
|-
| secrets.PAT || Token || Yes || Personal Access Token for repo write access
|-
| Issue body || Markdown || Yes || Issue template fields to parse
|}

''*One of comment or issue_number required depending on trigger type''

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Pull Request || GitHub PR || New PR with README.md changes
|-
| Commit || Git commit || "Add [App] to [Category] category"
|-
| Branch || String || add-{issue_number}
|-
| Closed Issue || GitHub Issue || Original issue closed with PR reference
|}

== Usage Examples ==

=== Example 1: Triggering via Comment ===
<syntaxhighlight lang="text">
1. Maintainer navigates to issue #42 (has "Add" label)
2. Maintainer comments: /convert
3. Workflow triggers
4. PR created: "Add Visual Studio Code to IDEs category"
5. Issue #42 closed with "Closes #42" in PR body
</syntaxhighlight>

=== Example 2: Triggering via Workflow Dispatch ===
<syntaxhighlight lang="text">
1. Navigate to Actions → "Convert Issue to Pull Request"
2. Click "Run workflow"
3. Enter issue_number: 42
4. Click "Run workflow"
5. Same result as comment trigger
</syntaxhighlight>

=== Example 3: Generated PR Content ===
<syntaxhighlight lang="markdown">
## Pull Request

**Title:** Add Visual Studio Code to IDEs category

**Body:**
This PR adds Visual Studio Code to the IDEs category in the README.md file.

Application URL: https://code.visualstudio.com
Repository URL: https://github.com/microsoft/vscode

Closes #42

**Diff:**
```diff
 ## IDEs

 * [Android Studio](https://developer.android.com/studio) - IDE for Android...
++ [Visual Studio Code](https://code.visualstudio.com/) - Lightweight but powerful source code editor. [![Open-Source Software][oss]](https://github.com/microsoft/vscode)
 * [Vim](https://vim.org) - Highly configurable text editor...
```
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:0PandaDEV_awesome-windows_PR_Review_Process]]

=== Requires Environment ===
* [[requires_env::Environment:0PandaDEV_awesome-windows_GitHub_Actions_Environment]]
