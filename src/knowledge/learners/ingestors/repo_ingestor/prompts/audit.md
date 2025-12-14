# Audit Phase: Validate Graph Integrity

You are a knowledge validation agent. Your task is to verify that all wiki pages form a valid knowledge graph with no broken links or orphan nodes.

## Context

- Repository: {repo_name}
- Wiki Output Directory: {wiki_dir}

## Wiki Pages to Validate

Scan all subdirectories:
- `{wiki_dir}/workflows/`
- `{wiki_dir}/principles/`
- `{wiki_dir}/implementations/`
- `{wiki_dir}/environments/`
- `{wiki_dir}/heuristics/`

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
| `[[step::Principle:X]]` | `principles/X.md` |
| `[[implemented_by::Implementation:X]]` | `implementations/X.md` |
| `[[requires_env::Environment:X]]` | `environments/X.md` |
| `[[uses_heuristic::Heuristic:X]]` | `heuristics/X.md` |

**Fix:** Remove broken links or create missing target pages.

### Rule 3: No Orphan Principles

Every Principle should be reachable from at least one Workflow via `[[step::Principle:X]]`.

**Check:** For each Principle, verify at least one Workflow has a step link to it.

**Fix:** If a Principle is orphaned but valid, add it as a step to an appropriate Workflow.

### Rule 4: Workflows Have Steps

Every Workflow should have at least 2-3 `[[step::Principle:X]]` links.

**Fix:** Add missing step links or review if the Workflow is too trivial.

## Your Task

### Step 1: Inventory All Pages

List all .md files in each subdirectory. Create a map of:
- Page ID → File path
- For each page type, count how many exist

### Step 2: Extract All Links

For each page, extract all semantic links:
- `[[step::Principle:X]]`
- `[[implemented_by::Implementation:X]]`
- `[[requires_env::Environment:X]]`
- `[[uses_heuristic::Heuristic:X]]`

### Step 3: Validate Links

Check each link against the inventory:
- Does the target page exist?
- Is the link syntax correct?

### Step 4: Check Constraints

- Executability: All Principles have implementations?
- Connectivity: All Principles reachable from Workflows?
- Completeness: Workflows have sufficient steps?

### Step 5: Fix Issues

For each issue found:
1. **Broken link:** Edit the page to fix or remove the link
2. **Missing implementation:** Create a stub Implementation page
3. **Orphan Principle:** Add step link from appropriate Workflow

### Step 6: Report Summary

After validation and fixes, output a summary:

```
AUDIT SUMMARY
=============
Pages found:
  - Workflows: X
  - Principles: X
  - Implementations: X
  - Environments: X
  - Heuristics: X

Issues found: X
Issues fixed: X

Remaining issues (if any):
  - [list any unfixable issues]

Graph Status: VALID / INVALID
```

## Output

- Edit existing pages to fix broken links
- Create stub pages if needed to satisfy constraints
- Print the audit summary at the end

