# Audit Phase: Validate Graph Integrity

You are a knowledge validation agent. Your task is to verify that all wiki pages form a valid knowledge graph with no broken links or orphan nodes.

## Context

- Repository: {repo_name}
- Wiki Output Directory: {wiki_dir}
- **Repository Map (Index):** {repo_map_path}
- **File Details:** {wiki_dir}/_files/

## IMPORTANT: Read Previous Phase Reports

**FIRST**, read the phase reports in `{wiki_dir}/_reports/`:
- `phase1_anchoring.md` - Workflows created
- `phase2_excavation_synthesis.md` - Implementation-Principle pairs created
- `phase3_enrichment.md` - Environments/Heuristics created, potential issues

These reports tell you what was created and any flagged issues.

## IMPORTANT: Use the Repository Map AND Page Indexes

Read the Repository Map at `{repo_map_path}` to:
- Check the **Coverage column** to understand what's documented
- Verify link targets reference real files/pages
- Identify gaps in coverage

**Also read the Page Indexes** for cross-reference validation:
- `{wiki_dir}/_WorkflowIndex.md` - Workflow pages
- `{wiki_dir}/_PrincipleIndex.md` - Principle pages
- `{wiki_dir}/_ImplementationIndex.md` - Implementation pages
- `{wiki_dir}/_EnvironmentIndex.md` - Environment pages
- `{wiki_dir}/_HeuristicIndex.md` - Heuristic pages

**Index Structure (all use same format):**
```
| Page | File | Connections | Notes |
|------|------|-------------|-------|
| {repo_name}_PageName | [→](./type/{repo_name}_PageName.md) | ✅Impl:{repo_name}_X, ⬜Principle:{repo_name}_Y | description |
```

**Key checks:**
- Every page file should have a row in its index
- Connections use **FULL names**: `✅Type:{repo_name}_Name` (exists) or `⬜Type:{repo_name}_Name` (missing)
- `⬜` references indicate pages that need creation

## Wiki Pages to Validate

Scan all subdirectories:
- `{wiki_dir}/workflows/`
- `{wiki_dir}/principles/`
- `{wiki_dir}/implementations/`
- `{wiki_dir}/environments/`
- `{wiki_dir}/heuristics/`

## Repo Scoping Rule (CRITICAL)

Only validate and fix pages whose filenames start with `{repo_name}_`.
Do NOT edit or create pages for other repos.

## Validation Rules

### Rule 1: Executability Constraint (CRITICAL)

**Every Principle MUST have at least one `[[implemented_by::Implementation:X]]` link.**

Check each file in `principles/`:
1. Find all `[[implemented_by::Implementation:X]]` links
2. Verify that file `implementations/X.md` exists
3. If no implementation link exists, this is a CRITICAL error

**Fix:** Either:
- Add the missing implementation link if an Implementation page exists
- Create a stub Implementation page if needed
- Or remove the Principle if it's not actually implemented

### Rule 2: Edge Targets Must Exist

All link targets must point to actual pages:

| Link Type | Target Directory |
|-----------|------------------|
| `[[step::Principle:{repo_name}_X]]` | `principles/{repo_name}_X.md` |
| `[[implemented_by::Implementation:{repo_name}_X]]` | `implementations/{repo_name}_X.md` |
| `[[requires_env::Environment:{repo_name}_X]]` | `environments/{repo_name}_X.md` |
| `[[uses_heuristic::Heuristic:{repo_name}_X]]` | `heuristics/{repo_name}_X.md` |

**Fix:** Remove broken links or create missing target pages.

### Rule 3: No Orphan Principles

Every Principle should be reachable from at least one Workflow via `[[step::Principle:X]]`.

**Fix:** If a Principle is orphaned but valid, add it as a step to an appropriate Workflow.

### Rule 4: Workflows Have Steps

Every Workflow should have at least 2-3 `[[step::Principle:X]]` links.

### Rule 5: Index Cross-References Are Valid

For each index file, verify that **Connections column** entries point to real pages:

- `✅Impl:{repo_name}_X` → `implementations/{repo_name}_X.md` must exist
- `✅Principle:{repo_name}_X` → `principles/{repo_name}_X.md` must exist
- `✅Env:{repo_name}_X` → `environments/{repo_name}_X.md` must exist
- `✅Heuristic:{repo_name}_X` → `heuristics/{repo_name}_X.md` must exist
- `✅Workflow:{repo_name}_X` → `workflows/{repo_name}_X.md` must exist

If a `✅` reference points to a non-existent page, either:
- Create the page, OR
- Change to `⬜` (if page genuinely doesn't exist)

### Rule 6: Indexes Match Directory Contents

**Every page file MUST have a row in its index:**
- Every `.md` in `workflows/` → row with `[→]` link in `_WorkflowIndex.md`
- Every `.md` in `principles/` → row with `[→]` link in `_PrincipleIndex.md`
- Every `.md` in `implementations/` → row with `[→]` link in `_ImplementationIndex.md`
- Every `.md` in `environments/` → row with `[→]` link in `_EnvironmentIndex.md`
- Every `.md` in `heuristics/` → row with `[→]` link in `_HeuristicIndex.md`

**Missing index entries are a CRITICAL error.** Add them.

### Rule 7: ⬜ References Need Resolution

Connections with `⬜` prefix indicate referenced-but-not-created pages.
- Either create the missing page and change `⬜Type:{repo_name}_Name` to `✅Type:{repo_name}_Name`
- Or remove the reference if it's no longer needed

Scan all index files for `⬜` and resolve them.
- Every `.md` file in `heuristics/` should have a row in `_HeuristicIndex.md`

## Your Task

### Step 1: Inventory All Pages via Indexes

**Read the 5 Page Index files** to inventory all pages:
1. `_WorkflowIndex.md` - List all Workflows
2. `_PrincipleIndex.md` - List all Principles
3. `_ImplementationIndex.md` - List all Implementations
4. `_EnvironmentIndex.md` - List all Environments
5. `_HeuristicIndex.md` - List all Heuristics

Also scan the actual directories to verify indexes match reality:
- Every page in directory should have an index entry
- Every index entry should have a corresponding page file

### Step 2: Extract All Links

For each page, extract all semantic links.

### Step 3: Validate Links

Check each link against the inventory.

### Step 4: Check Constraints

- Executability: All Principles have implementations?
- Connectivity: All Principles reachable from Workflows?
- Completeness: Workflows have sufficient steps?

### Step 5: Fix Issues

For each broken link:
1. **FIRST CHOICE: Remove the broken link**
2. **SECOND CHOICE: Create the missing target page** with real content

For missing `implemented_by` links:
- Find an existing Implementation that implements this Principle
- Add the link

For orphan Principles:
- Find a Workflow that logically includes this Principle
- Add a step link

### Step 6: Update Repository Map Coverage

Ensure the **Coverage column** in `{repo_map_path}` accurately reflects what's documented.

If a file now has more coverage (new pages created), update it:
```markdown
| ✅ | `unsloth/save.py` | 450 | Model saving | Impl: save_pretrained, save_gguf; Env: — | [→](...) |
```

### Step 7: Update Missing Index Entries

For any pages missing from indexes, add the appropriate row.

### Step 8: Report Summary

Output a summary:

```
AUDIT SUMMARY
=============
Pages found:
  - Workflows: X
  - Principles: X
  - Implementations: X
  - Environments: X
  - Heuristics: X

Index Validation:
  - Missing index entries fixed: X
  - Invalid cross-references fixed: X

Link Issues:
  - Issues found: X
  - Issues fixed: X

Remaining issues (if any):
  - [list any unfixable issues]

Graph Status: VALID / INVALID
```

## Output

- Edit existing pages to fix broken links
- Create stub pages if needed
- Update Repository Map coverage
- Print the audit summary

## 📝 Execution Report (REQUIRED)

When finished, write a summary report to `{wiki_dir}/_reports/phase4_audit.md`:

```markdown
# Phase 5: Audit Report

## Graph Statistics
| Type | Count |
|------|-------|
| Workflows | X |
| Principles | X |
| Implementations | X |
| Environments | X |
| Heuristics | X |

## Issues Fixed
- Broken links removed: X
- Missing pages created: X
- Missing index entries added: X

## Remaining Issues
- [Any unfixed issues]

## Graph Status: VALID / NEEDS_ATTENTION

## Notes for Orphan Mining Phase
- [Files with Coverage: — that should be checked]
- [Uncovered areas of the codebase]
```
