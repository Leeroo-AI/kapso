# Orphan Mining Phase: Inventory Scan for Uncaptured Code

You are a Code Archivist. Your task is to systematically map all valid code units that were NOT captured during the Workflow-based analysis.

## Context

- Repository: {repo_name}
- Repository Path: {repo_path}
- Wiki Output Directory: {wiki_dir}
- **Repository Map (Index):** {repo_map_path}
- **File Details:** {wiki_dir}/_files/

**Already Mapped Pages:**
- Workflows: {wiki_dir}/workflows/
- Principles: {wiki_dir}/principles/
- Implementations: {wiki_dir}/implementations/

## IMPORTANT: Read Previous Phase Reports

**FIRST**, read the Audit report at `{wiki_dir}/_reports/phase5_audit.md`.

This report tells you:
- Current graph statistics
- Files flagged as uncovered
- Areas needing orphan mining

## CRITICAL: Use the Repository Map for Coverage

**THEN**, read the Repository Map index at `{repo_map_path}`.

The **Coverage column** tells you EXACTLY which files are already documented:
- `—` means NOT covered (orphan candidate)
- `Impl: X` means covered by Implementation page X
- `Workflow: Y` means covered by Workflow page Y

**Focus on files where Coverage is `—` or incomplete.**

## CRITICAL: Check Page Indexes for Missing Pages

**Also scan ALL index files for `⬜` references:**
- `{wiki_dir}/_WorkflowIndex.md`
- `{wiki_dir}/_ImplementationIndex.md`
- `{wiki_dir}/_PrincipleIndex.md`
- `{wiki_dir}/_EnvironmentIndex.md`
- `{wiki_dir}/_HeuristicIndex.md`

**How to read the Connections column:**
- `✅Type:{repo_name}_Name` = Page EXISTS (already created)
- `⬜Type:{repo_name}_Name` = Page MISSING (orphan candidate!)

**Your job:** Create pages for `⬜` references that map to real code.

## Repo Scoping Rule (CRITICAL)

Only consider pages whose filenames start with `{repo_name}_` as "Already Mapped".
Only create/update pages whose filenames start with `{repo_name}_`.

## Your Task: Find the "Dark Matter"

### Step 1: Identify Orphan Files from Index

Read `{repo_map_path}` and find all files where:
- **Coverage column is `—`** (not covered at all)
- **Coverage is partial** (e.g., only has Workflow but no Impl)

These are your **Candidate Files**.

### Step 2: The Significance Filter (Discard Noise)

For each Candidate File with `Coverage: —`:

**Criterion A: Structure**
- Does the file contain a Public Class or Major Public Function?
- Check the file's detail page in `_files/` for Classes/Functions
- If NO (only `_private_funcs`, configs, string utils) → **DISCARD**

**Criterion B: Test Coverage (Proxy Check)**
- Is there a test file for this component?
- If YES → High confidence, **KEEP**
- If NO → Read the code. Does it perform a distinct algorithmic task?

### Step 3: Implementation Extraction

For every file that passed the filter:

1. **Read** the source file from `{repo_path}`
2. **Read** the file's detail page from `_files/` for context
3. **Create** an Implementation page with:
   - **Metadata block** (wikitable with sources, domains, last_updated)
     - ⚠️ Sources must be HIGH-LEVEL: repo URLs, docs, papers (NOT file paths!)
     - ❌ WRONG: `[[source::Repo|Loader|unsloth/models/loader.py]]`
     - ✅ RIGHT: `[[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]`
   - `== Code Reference ==` with (file paths go HERE):
     - Source Location (GitHub URL with line numbers)
     - Full Signature
     - Import statement
   - `== I/O Contract ==` as structured tables
   - `== Usage Examples ==` with runnable code

**Code Reference Example:**
```mediawiki
=== Source Location ===
* '''File:''' [{repo_url}/blob/main/path/file.py#L10-L50 path/file.py]
* '''Lines:''' 10-50

=== Signature ===
<syntaxhighlight lang="python">
def orphan_function(input: Tensor) -> Tensor:
    """Process input tensor."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from package.module import orphan_function
</syntaxhighlight>
```

**Output:** Write to `{wiki_dir}/implementations/`
**Filename format:** `{repo_name}_OrphanClassName.md`

### Step 4: Principle Synthesis & Polymorphism Check

**CRITICAL:** Before creating a new Principle, check existing Principles!

**Question:** "Do I already have a Principle that describes WHAT this code does?"

**Path A: It is a Variant (Polymorphism)**
- Link new Implementation to the EXISTING Principle
- Update the Principle's Related Pages

**Path B: It is Unique**
- Create a NEW Principle page with specific name
- Link it to the Implementation

### Step 5: Update Coverage in Repository Map

After creating pages for orphan files, **update the index** at `{repo_map_path}`:

```markdown
| ✅ | `unsloth/kernels/geglu.py` | 80 | GEGLU activation | Impl: geglu_kernel; Principle: Gated_Activation | [→](...) |
```

**This is critical for tracking progress** — change Coverage from `—` to the actual pages.

### Step 6: Context Mining (Heuristics & Environment)

For orphan code, capture constraints:

1. **Scan** for `raise Error`, `warnings.warn`, `assert`
2. **Create** Heuristic pages for discovered wisdom
3. **Link** `[[uses_heuristic::Heuristic:X]]` in the Implementation

4. **Scan** for imports or hardware checks
5. **Create** Environment pages if new requirements found
6. **Link** `[[requires_env::Environment:X]]`

### Step 7: Update Page Indexes (IMMEDIATELY)

**⚠️ CRITICAL:** Update indexes **IMMEDIATELY after creating each orphan page**.

Use **FULL page names** in Connections column: `✅Type:{repo_name}_Name` (exists) or `⬜Type:{repo_name}_Name` (not created).

**For new Implementations** → Add to `{wiki_dir}/_ImplementationIndex.md`:
```
| {repo_name}_OrphanClass | [→](./implementations/...) | ⬜Principle:{repo_name}_X, ⬜Env:{repo_name}_Y | file.py:L10-50 - Description |
```

**For new Principles** → Add to `{wiki_dir}/_PrincipleIndex.md`:
```
| {repo_name}_OrphanPrinciple | [→](./principles/...) | ✅Impl:{repo_name}_OrphanClass | Theoretical concept |
```

**For new Environments** → Add to `{wiki_dir}/_EnvironmentIndex.md`:
```
| {repo_name}_NewEnv | [→](./environments/...) | ✅Impl:{repo_name}_OrphanClass | Requirements |
```

**For new Heuristics** → Add to `{wiki_dir}/_HeuristicIndex.md`:
```
| {repo_name}_NewHeuristic | [→](./heuristics/...) | ✅Impl:{repo_name}_OrphanClass | Tips |
```

Each connection shows whether that referenced page exists (✅) or needs creation (⬜).

### Step 8: Update Other Indexes (Bi-directional)

When you create orphan pages, also update references in OTHER indexes:

1. **Search ALL index files** for `⬜Type:{repo_name}_YourPageName`
2. **Change to `✅Type:{repo_name}_YourPageName`** wherever you find it

**Example:** If you create `{repo_name}_OrphanKernel` Implementation:
```markdown
# Search _WorkflowIndex.md, _PrincipleIndex.md for:
⬜Impl:{repo_name}_OrphanKernel

# Change to:
✅Impl:{repo_name}_OrphanKernel
```

This ensures cross-references stay accurate across all indexes.

## Wiki Structure Definitions

{implementation_structure}

{principle_structure}

## ⚠️ File Editing Tip

When updating index files:
- **Use Write tool** (read entire file → modify → write back)
- **Avoid Edit tool** — it often fails on markdown tables

## 📝 Execution Report (REQUIRED)

When finished, write a summary report to `{wiki_dir}/_reports/phase6_orphan_mining.md`:

```markdown
# Phase 6: Orphan Mining Report

## Scan Summary
- Files scanned: X
- Files with existing coverage: X
- Orphan candidates found: X

## Pages Created
| Type | Page | Source File |
|------|------|-------------|
| Implementation | [name] | [file] |
| Principle | [name] | [linked impl] |

## Decisions Made
- Discarded as noise: X files
- Linked to existing Principles: X
- New Principles created: X

## Notes for Orphan Audit Phase
- [Pages that need hidden workflow check]
- [Potential deprecated code]
- [Names that may be too generic]
```
