# Principle: Issue_Body_Parsing

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|GitHub Script Action|https://github.com/actions/github-script]]
* [[source::Doc|awk Manual|https://www.gnu.org/software/gawk/manual/gawk.html]]
|-
! Domains
| [[domain::Text_Parsing]], [[domain::Data_Extraction]], [[domain::GitHub_Actions]]
|-
! Last Updated
| [[last_updated::2026-01-08 18:30 GMT]]
|}

== Overview ==

Technique for extracting structured data from GitHub Issue bodies into usable variables.

=== Description ===

Issue Body Parsing is the process of extracting field values from GitHub Issue bodies that were created using Issue Forms. Since Issue Forms produce predictable markdown output with `### Header` delimiters, text processing tools like awk can reliably extract values. This enables automation workflows to transform unstructured issue text into structured data for processing.

The two-phase approach combines GitHub API access (to fetch issue content) with text processing tools (to parse the content), separating concerns and using appropriate tools for each task.

=== Usage ===

Use this principle when:
- Processing user submissions from Issue Forms
- Automating issue-to-PR conversion workflows
- Extracting metadata from issues for labeling or routing
- Building data pipelines from GitHub Issues

== Theoretical Basis ==

=== Parsing Strategy ===
<syntaxhighlight lang="text">
Issue Forms Output Structure:
### Field Name 1
value1

### Field Name 2
value2

Parsing Algorithm:
1. Find header line matching target field
2. Capture all lines until next header
3. Trim whitespace from result
</syntaxhighlight>

=== awk Field Extraction ===
<syntaxhighlight lang="awk">
# Pattern: /### Field Name/{flag=1; next} /###/{flag=0} flag

# State machine:
# - When "### Field Name" found: enable capture, skip header line
# - When any "###" found: disable capture
# - Otherwise: if capture enabled, print line

# Compact single-line form:
awk '/### Field Name/{f=1;next}/###/{f=0}f'

# With xargs for whitespace trimming:
echo "$body" | awk '/### Field Name/{f=1;next}/###/{f=0}f' | xargs
</syntaxhighlight>

=== API + Shell Pipeline ===
<syntaxhighlight lang="text">
Phase 1: Fetch (JavaScript/GitHub Script)
  - Use github.rest.issues.get()
  - Return issue.data object
  - Access via steps.id.outputs.result

Phase 2: Parse (Shell/awk)
  - Receive body as string
  - Apply awk patterns per field
  - Store results in environment variables
</syntaxhighlight>

=== Multi-field Extraction ===
<syntaxhighlight lang="bash">
ISSUE_BODY="${{ fromJson(steps.issue.outputs.result).body }}"

APP_NAME=$(echo "$ISSUE_BODY" | awk '/### Application Name/{f=1;next}/###/{f=0}f' | xargs)
APP_URL=$(echo "$ISSUE_BODY" | awk '/### Application URL/{f=1;next}/###/{f=0}f' | xargs)
CATEGORY=$(echo "$ISSUE_BODY" | awk '/### Category/{f=1;next}/###/{f=0}f' | xargs)

# Each field uses identical pattern, just different header name
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:0PandaDEV_Awesome_windows_issue_metadata_extraction]]
