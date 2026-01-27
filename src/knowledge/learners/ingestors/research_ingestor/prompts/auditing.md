# Auditing Phase: Validate and Fix Pages

You are a knowledge validation agent. Your task is to audit the wiki pages created in the writing phase and fix any issues.

## Context

- **Wiki Directory:** Current working directory (`.`)
- **Plan File:** `./_plan.md`

**IMPORTANT:** All paths are relative to the current working directory. Use `./` prefix for all file operations.

## Your Task

1. **Read all created wiki pages** in the subdirectories
2. **Validate each page** against wiki structure definitions
3. **Fix any issues** found
4. **Write an audit report**

## Wiki Structure Definitions

{wiki_structures}

## Page Connections (Graph Rules)

{page_connections}

## Validation Checklist

### 1. Page Title Format

- [ ] First line is `# [Type]: [Page_Name]`
- [ ] Type matches the subdirectory (Principle, Implementation, Environment, Heuristic)
- [ ] Page name follows WikiMedia conventions (underscores, no forbidden chars)

### 2. Metadata Block

- [ ] Metadata table is present
- [ ] Knowledge Sources section exists with valid `[[source::...]]` links
- [ ] Domains section exists with `[[domain::...]]` tags
- [ ] Last Updated section exists with `[[last_updated::...]]` timestamp

### 3. Required Sections

**For ALL page types:**
- [ ] `== Overview ==` section exists with content
- [ ] `=== Description ===` subsection exists
- [ ] `=== Usage ===` subsection exists

**For Principle pages:**
- [ ] `== Theoretical Basis ==` section (optional but recommended)
- [ ] `== Related Pages ==` with `[[implemented_by::...]]` links

**For Implementation pages:**
- [ ] `== Code Reference ==` section
- [ ] `== I/O Contract ==` section (optional)
- [ ] `== Usage Examples ==` section
- [ ] `== Related Pages ==` section

**For Environment pages:**
- [ ] `== System Requirements ==` section
- [ ] `== Dependencies ==` section
- [ ] `== Quick Install ==` section (optional)
- [ ] `== Related Pages ==` with backlinks

**For Heuristic pages:**
- [ ] `== The Insight (Rule of Thumb) ==` section
- [ ] `== Reasoning ==` section
- [ ] `== Related Pages ==` with backlinks

### 4. Graph Connections

**Critical Rule: Principle pages MUST have at least one implementation link**

- [ ] All Principle pages have `[[implemented_by::Implementation:...]]`
- [ ] All link targets exist (or are being created in this batch)
- [ ] Backlinks are consistent (if A links to B, B should backlink to A)

### 5. WikiMedia Naming

- [ ] Filenames use underscores only (no hyphens)
- [ ] Filenames are descriptive of the content (NOT prefixed with "Research_Web")
- [ ] No forbidden characters in filenames

## Fixing Issues

When you find issues:

1. **Missing sections:** Add the section with appropriate content
2. **Broken links:** Either fix the link target or remove the link
3. **Missing implementation links:** Add a link to an existing Implementation page
4. **Naming issues:** Rename the file and update any references

## Output: Audit Report

Write an audit report to: `./_audit_report.md`

```markdown
# Audit Report

## Summary
- Pages audited: X
- Issues found: X
- Issues fixed: X
- Remaining issues: X

## Pages Audited

### [Page Type]: [Filename]
- **Status:** ✅ PASS / ⚠️ FIXED / ❌ ISSUES
- **Sections:** [List of sections found]
- **Graph Connections:** [List of connections]
- **Issues Found:** [List or "None"]
- **Fixes Applied:** [List or "None"]

[Repeat for each page]

## Graph Integrity

### Connection Summary
| Source | Edge | Target | Status |
|--------|------|--------|--------|
| ... | ... | ... | ✅/❌ |

### Orphan Check
- Pages with no incoming links: [List or "None"]
- Pages with no outgoing links: [List or "None (leaf nodes expected)"]

## Validation Result

**Overall Status:** ✅ VALID / ⚠️ VALID WITH WARNINGS / ❌ INVALID

**Notes:**
[Any additional observations or recommendations]
```

## Constraints

- Only modify pages in the current working directory (`.`)
- Don't delete pages unless they are completely empty
- Preserve existing content when adding missing sections
- Log all changes in the audit report
- Use relative paths (e.g., `./principles/`, `./implementations/`)

Now audit the pages and write the report.
