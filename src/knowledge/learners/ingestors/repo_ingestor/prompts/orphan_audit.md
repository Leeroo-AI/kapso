# Orphan Audit Phase: Quality Control for Orphan Nodes

You are a Quality Auditor. Your task is to validate that orphan nodes (created in the Orphan Mining phase) are truly orphans and are actionable.

## ⚠️ FILE PLACEMENT RULES (CRITICAL)

**Only modify/create files in these directories:**
- `{wiki_dir}/principles/` - Principle pages (rename if needed)
- `{wiki_dir}/implementations/` - Implementation pages (rename if needed)
- `{wiki_dir}/heuristics/` - Deprecation warnings
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
✅ {repo_name}_Hidden_API_Name
❌ {repo_name}_Hidden-API       (hyphen)
```

## Context

- Repository: {repo_name}
- Repository Path: {repo_path}
- Wiki Output Directory: {wiki_dir}
- **Repository Map (Index):** {repo_map_path}
- **File Details:** {wiki_dir}/_files/

## IMPORTANT: Read Previous Phase Reports

**FIRST**, read the Orphan Mining reports:
- `{wiki_dir}/_reports/phase5b_orphan_review.md` — MANUAL_REVIEW decisions
- `{wiki_dir}/_reports/phase5c_orphan_create.md` — Pages created, coverage updates

Also check `{wiki_dir}/_orphan_candidates.md` to see:
- AUTO_KEEP files (all should have `✅ DONE` status)
- MANUAL_REVIEW files (should have `✅ APPROVED` or `❌ REJECTED`)

These reports tell you:
- Which orphan pages were created
- Potential naming issues

## IMPORTANT: Use the Repository Map AND Page Indexes

Read the Repository Map at `{repo_map_path}` to:
- Check **Coverage column** to see what's marked as covered
- Verify coverage accuracy

**Also read the Page Indexes** for cross-reference validation:
- `{wiki_dir}/_PrincipleIndex.md` - Verify orphan Principles are listed
- `{wiki_dir}/_ImplementationIndex.md` - Verify orphan Implementations are listed

**Index Structure (all use same format):**
```
| Page | File | Connections | Notes |
|------|------|-------------|-------|
| {repo_name}_PageName | [→](./type/...) | ✅Impl:{repo_name}_X, ⬜Principle:{repo_name}_Y | description |
```

**Key checks:**
- All orphan pages from Phase 6 should have rows with `[→]` links
- Connections use **FULL names**: `✅Type:{repo_name}_Name` (exists) or `⬜Type:{repo_name}_Name` (missing)
- `⬜` references indicate pages still needing creation

## Your Task: Validate Orphan Nodes

Perform these checks on ALL Implementation and Principle pages, especially those created in the Orphan Mining phase.

## Repo Scoping Rule (CRITICAL)

Only validate and fix pages whose filenames start with `{repo_name}_`.

---

### Check 1: The "Dead Code" Check

**Goal:** Identify deprecated or legacy code.

**Action:**
1. Scan the source file for `@deprecated` decorators
2. Check if the file is in `legacy/`, `old/`, `deprecated/` directory
3. Look for comments like `# TODO: remove`, `# DEPRECATED`

**Decision:**
- **If Deprecated/Legacy:**
  - Create a Heuristic: `{repo_name}_Warning_Deprecated_X`
  - Link it to the Implementation
  - Add warning in the Implementation's Description
  - **Update Coverage** to note deprecation
  
- **If Not Deprecated:** Proceed to Check 2.

---

### Check 2: The "Naming Specificity" Check

**Goal:** Ensure orphan nodes are self-descriptive.

**Action:**
Review all Principle names, especially newly created ones.

**Bad Names (too generic):**
- "Optimization", "Processing", "Utility", "Helper"

**Good Names (specific):**
- "Gradient_Checkpointing_Optimization"
- "RMS_Normalization"
- "Triton_Fused_Attention_Kernel"

**Decision:**
- **If Name is Generic:**
  - Rename to be implementation-specific
  - Update all links to use the new name
  - A floating node MUST be self-descriptive

---

### Check 3: Verify Repository Map Coverage

**Goal:** Ensure the Repository Map accurately reflects all coverage.

**Action:**
1. For each file in the index with `Coverage: —`, verify no pages exist for it
2. For each file with coverage listed, verify those pages actually exist
3. Fix any mismatches

### Check 4: Verify Page Index Completeness

**Goal:** Ensure all pages are listed in their indexes with correct connection statuses.

**Action:**
1. For each `.md` file in `implementations/`:
   - Must have a row in `_ImplementationIndex.md` with `[→]` link
   - Connections should match page's Related Pages section
   - Each connection should have correct status: `✅` if page exists, `⬜` if not
2. For each `.md` file in `principles/`:
   - Must have a row in `_PrincipleIndex.md` with `[→]` link
3. For each `⬜Type:{repo_name}_Name` reference in any index:
   - Either create the missing page and change to `✅Type:{repo_name}_Name`
   - Or remove the reference if no longer needed
4. Verify all `✅Type:{repo_name}_Name` references point to real pages

---

## Final Output

### Update Repository Map

Ensure all coverage changes are reflected in `{repo_map_path}`.

### Report Summary

```
ORPHAN AUDIT SUMMARY
====================
Total orphan Implementations checked: X
Total orphan Principles checked: X

Deprecated code flagged: X
  - [list any deprecation warnings added]

Names corrected: X
  - [list any renames]

Index Updates:
  - Missing ImplementationIndex entries added: X
  - Missing PrincipleIndex entries added: X
  - Invalid cross-references fixed: X

Coverage column corrections: X
  - [list any fixes to the index]

Orphan Status:
  - Confirmed orphans: X
  - Flagged as deprecated: X
```

## 📝 Final Execution Report (REQUIRED)

When finished, write the final summary to `{wiki_dir}/_reports/phase6_orphan_audit.md`:

```markdown
# Phase 6: Orphan Audit Report (FINAL)

## Final Graph Statistics
| Type | Count |
|------|-------|
| Workflows | X |
| Principles | X |
| Implementations | X |
| Environments | X |
| Heuristics | X |

## Orphan Audit Results
- Deprecated code flagged: X
- Names corrected: X
- Index entries fixed: X

## Final Status
- Confirmed orphans: X
- Total coverage: X% of source files

## Graph Integrity: ✅ VALID / ⚠️ NEEDS REVIEW

## Summary
[Brief summary of the entire ingestion process and final knowledge graph quality]
```

## Wiki Structure Definitions

{implementation_structure}

{principle_structure}

{heuristic_structure}
