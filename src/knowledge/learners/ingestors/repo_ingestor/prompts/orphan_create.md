# Phase 6c: Orphan Page Creation

You are a Knowledge Extractor. Your task is to create wiki pages for all approved orphan files.

## ⚠️ FILE PLACEMENT RULES (CRITICAL)

**Only create files in these directories:**
- `{wiki_dir}/implementations/` - Implementation pages
- `{wiki_dir}/principles/` - Principle pages
- `{wiki_dir}/_reports/` - Execution reports

**DO NOT create:**
- Summary files at the root of `{wiki_dir}`
- Documentation files outside the designated directories
- Any file that doesn't follow the `{repo_name}_Page_Name.md` naming convention
- "Notes", "summaries", or "completion reports" outside `_reports/`

## 📛 PAGE NAMING RULES (WikiMedia Compliance)

All page names must follow WikiMedia technical syntax:

### Syntax Rules
1. **First letter capitalized** — System auto-converts
2. **Underscores only** — Use `_` as the sole word separator (NO hyphens, NO spaces)
3. **Case-sensitive after first character**

### Forbidden Characters (NEVER use)

`#`, `<`, `>`, `[`, `]`, `{{`, `}}`, `|`, `+`, `:`, `/`, `-` (hyphen)

### Examples

```
✅ {repo_name}_ClassName_Method
❌ {repo_name}_ClassName-Method   (hyphen)
❌ {repo_name}_Class/Method       (slash)
```

## Context

- Repository: {repo_name}
- Repository Path: {repo_path}
- Wiki Output Directory: {wiki_dir}
- **Repository Map (Index):** {repo_map_path}
- **Orphan Candidates:** {candidates_path}
- **File Details:** {wiki_dir}/_files/

## Approved Files

Read `{candidates_path}` to get the list of files that need wiki pages:

1. **ALL files in AUTO_KEEP section** — These MUST be documented (no exceptions)
2. **Files in MANUAL_REVIEW with `✅ APPROVED`** — These were approved by the review step

## Wiki Structure Definitions

### Implementation Page Structure

{implementation_structure}

### Principle Page Structure

{principle_structure}

---

## Your Task

### Step 1: Get the Approved File List

Read `{candidates_path}` and collect:
- All files from AUTO_KEEP section (regardless of status)
- All files from MANUAL_REVIEW with `✅ APPROVED` decision

### Step 2: Create Pages for Each Approved File

For EACH approved file:

1. **Read the source file** from `{repo_path}`
2. **Read the file's detail page** in `{wiki_dir}/_files/` for context
3. **Create an Implementation page** following the structure above

#### Implementation Page Requirements

Each Implementation page MUST include:

**Metadata block:**
```mediawiki
{{{{wikitable
| Sources = [[source::Repo|{repo_name}|{repo_url}]]
| Domains = [[domain::X]], [[domain::Y]]
| Last Updated = {{date}}
}}}}
```

**== Overview ==**
One sentence describing what this code does.

**== Code Reference ==**
```mediawiki
=== Source Location ===
* '''Repository:''' [{repo_url} {repo_name}]
* '''File:''' [{repo_url}/blob/main/path/file.py#L10-L100 path/file.py]
* '''Lines:''' 10-100

=== Signature ===
<syntaxhighlight lang="python">
class ClassName:
    def method(self, arg: Type) -> ReturnType:
        """Docstring."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from package.module import ClassName
</syntaxhighlight>
```

**== I/O Contract ==**
Tables for inputs and outputs.

**== Usage Examples ==**
Complete, runnable code examples.

**== Related Pages ==**
Links to Principles, Environments, Heuristics.

### Step 3: Principle Synthesis (Polymorphism Check)

**CRITICAL:** Before creating a new Principle, check existing Principles!

Ask: "Do I already have a Principle that describes WHAT this code does?"

**If YES (Variant/Polymorphism):**
- Link new Implementation to the EXISTING Principle
- Update the Principle's Related Pages section
- Add `[[implemented_by::Implementation:{repo_name}_NewImpl]]`

**If NO (Unique concept):**
- Create a NEW Principle page with specific name
- Link it to the Implementation

### Step 4: Checkpoint Progress (CRITICAL)

After creating EACH page, **IMMEDIATELY update the Status column** in `{candidates_path}`:

```markdown
## AUTO_KEEP (Must Document)
| # | File | Lines | Rule | Status |
|---|------|-------|------|--------|
| 1 | `path/file.py` | 500 | K1 | ✅ DONE |  ← Update this
```

This allows resumption if interrupted.

### Step 5: Update Repository Map Coverage

After creating pages, **update the Coverage column** in `{repo_map_path}`:

```markdown
| ✅ | `path/file.py` | 500 | Purpose | Impl: {repo_name}_ClassName | [→](...) |
```

If a Principle was also created:
```markdown
| ✅ | `path/file.py` | 500 | Purpose | Impl: {repo_name}_X; Principle: {repo_name}_Y | [→](...) |
```

### Step 6: Update Page Indexes (IMMEDIATELY)

**⚠️ CRITICAL:** Update indexes **IMMEDIATELY after creating each page**.

**For new Implementations** → Add to `{wiki_dir}/_ImplementationIndex.md`:
```markdown
| {repo_name}_ClassName | [→](./implementations/{repo_name}_ClassName.md) | ⬜Principle:{repo_name}_X | file.py:L10-100 |
```

**For new Principles** → Add to `{wiki_dir}/_PrincipleIndex.md`:
```markdown
| {repo_name}_ConceptName | [→](./principles/{repo_name}_ConceptName.md) | ✅Impl:{repo_name}_ClassName | Theoretical concept |
```

### Step 7: Update Cross-References (Bi-directional)

When you create a page, search ALL index files for `⬜Type:{repo_name}_YourPageName` and change to `✅Type:{repo_name}_YourPageName`.

---

## Output Locations

- Implementation pages: `{wiki_dir}/implementations/{repo_name}_Name.md`
- Principle pages: `{wiki_dir}/principles/{repo_name}_Name.md`

## Repo Scoping Rule (CRITICAL)

Only create pages whose filenames start with `{repo_name}_`.

## ⚠️ File Editing Tip

When updating index files or candidates file:
- **Use Write tool** (read entire file → modify → write back)
- **Avoid Edit tool** — it often fails on markdown tables

## Completion Criteria

Task is complete when:
- [ ] ALL AUTO_KEEP files have `✅ DONE` status
- [ ] ALL APPROVED MANUAL_REVIEW files have wiki pages
- [ ] RepoMap Coverage column is updated for all documented files
- [ ] Page indexes are updated

## 📝 Execution Report (REQUIRED)

When finished, write a summary report to `{wiki_dir}/_reports/phase5c_orphan_create.md`:

```markdown
# Phase 6c: Orphan Page Creation Report

## Pages Created

### Implementations
| Page | Source File | Lines |
|------|-------------|-------|
| {repo_name}_X | path/file.py | 10-100 |

### Principles
| Page | Implemented By |
|------|----------------|
| {repo_name}_Concept | {repo_name}_X |

## Summary
- Implementation pages created: X
- Principle pages created: X
- Files linked to existing Principles: X

## Coverage Updates
- RepoMap entries updated: X
- Index entries added: X

## Notes for Orphan Audit Phase
- [Pages that may need hidden workflow check]
- [Potential naming improvements]
```

