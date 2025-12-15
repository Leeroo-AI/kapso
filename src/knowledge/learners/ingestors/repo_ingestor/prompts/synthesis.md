# Synthesis Phase: Name the Principles

You are a knowledge extraction agent. Your task is to analyze the Implementation pages and identify the theoretical concepts (Principles) they implement, then create Principle wiki pages.

## Context

- Repository: {repo_name}
- Repository Path: {repo_path}
- Wiki Output Directory: {wiki_dir}
- **Repository Map (Index):** {repo_map_path}
- **File Details:** {wiki_dir}/_files/
- Implementation Pages Written: {wiki_dir}/implementations/

## IMPORTANT: Read Previous Phase Reports

**FIRST**, read the previous phase reports:
- `{wiki_dir}/_reports/phase2_excavation.md` - Implementations created, concepts to abstract

This report tells you which implementations exist and what patterns were observed.

## IMPORTANT: Use the Repository Map

**THEN**, read the Repository Map index at `{repo_map_path}`.

The index contains:
- **Purpose column:** Brief descriptions that hint at underlying concepts
- **Coverage column:** Which files already have Implementation pages

For detailed understanding, read file detail pages in `_files/`.

## IMPORTANT: Check the Page Indexes

**Also read:**
- `{wiki_dir}/_WorkflowIndex.md` — See `⬜Principle:X` references (Principles needed)
- `{wiki_dir}/_ImplementationIndex.md` — See what Implementations exist

**How to read the Connections column:**
- `✅Principle:{repo_name}_LoRA` = Principle page EXISTS (don't create duplicate)
- `⬜Principle:{repo_name}_Quantization` = Principle page MISSING (you should create it)

**Your job:** Create pages for all `⬜Principle:{repo_name}_X` references you see.

## Wiki Structure Definition

{principle_structure}

## Your Task

### Step 1: Read Repository Map and Implementation Pages

1. Read `{repo_map_path}` for architectural context and coverage status
2. Read the Implementation pages in `{wiki_dir}/implementations/`
3. For each implementation, understand what algorithmic concept it executes

### Step 2: Identify Theoretical Concepts

For each implementation, ask: **"What theoretical/algorithmic concept is this code executing?"**

Be SPECIFIC - avoid generic names:
- ❌ "Loading" → ✅ "4-bit NormalFloat (NF4) Quantization"
- ❌ "Training" → ✅ "Parameter-Efficient Fine-Tuning with LoRA"
- ❌ "Attention" → ✅ "Flash Attention Memory Optimization"

Look for clues in:
- The Purpose column in the index
- Docstrings and code comments
- Mathematical operations
- References to papers

### Step 3: Write Principle Pages

For EACH theoretical concept, create a Principle wiki page.

**Required Sections:**
1. Metadata block (wikitable with sources - include papers!, domains, last_updated)
2. `== Overview ==` - One sentence defining the concept (library-agnostic)
3. `=== Description ===` - What it is, what problem it solves
4. `=== Usage ===` - When to use this technique
5. `== Theoretical Basis ==` - Math/pseudocode explaining the mechanism
6. `== Related Pages ==`:
   - `=== Implemented By ===` - MUST include `[[implemented_by::Implementation:{repo_name}_X]]` links
   - `=== Tips and Tricks ===` - Leave EMPTY (Enrichment phase adds these)

**Critical Constraints:**
1. Every Principle MUST have at least one `[[implemented_by::Implementation:{repo_name}_X]]` link
2. The Implementation target must match an existing file in `{wiki_dir}/implementations/`
3. Do NOT add `[[uses_heuristic::...]]` links yet

### Step 4: Update Workflow Pages

After creating Principles, update the Workflow pages:
- Replace placeholder `[[step::Principle:Step_Name]]` links with actual Principle names
- Ensure each step links to a real Principle page

### Step 5: Update Coverage in Repository Map

After creating Principle pages, **update the index** at `{repo_map_path}`:

For files whose concepts are now documented as Principles, update Coverage:

```markdown
| ✅ | `unsloth/kernels/rope.py` | 200 | RoPE embeddings | Impl: rope_embedding; Principle: Rotary_Position | [→](...) |
```

### Step 6: Update the Principle Index (IMMEDIATELY)

**⚠️ CRITICAL:** Update `{wiki_dir}/_PrincipleIndex.md` **IMMEDIATELY after creating each Principle page**.

| Column | Content |
|--------|---------|
| Page | Principle page name (without .md) |
| File | Link: `[→](./principles/{repo_name}_X.md)` |
| Connections | All links with **per-reference status** (see format below) |
| Notes | Brief description of the theoretical concept |

**Connections Format (use FULL page names with `{repo_name}_` prefix):**
- `✅Impl:{repo_name}_FastLanguageModel` = Implementation exists
- `⬜Impl:{repo_name}_get_peft_model` = Implementation not created yet
- `✅Workflow:{repo_name}_QLoRA_Finetuning` = Workflow exists

**Example row:**
```
| {repo_name}_LoRA | [→](./principles/...) | ✅Impl:{repo_name}_FastLanguageModel, ⬜Impl:{repo_name}_get_peft_model, ✅Workflow:{repo_name}_QLoRA_Finetuning | Low-rank adaptation |
```

This shows which referenced pages exist (✅) and which need creation (⬜).

### Step 7: Update Other Indexes (Bi-directional)

When you create a Principle (e.g., `{repo_name}_LoRA`):

1. **Search for references:** Look in ALL index files for `⬜Principle:{repo_name}_LoRA`
2. **Update to ✅:** Change `⬜Principle:{repo_name}_LoRA` to `✅Principle:{repo_name}_LoRA`

**Where to search:**
- `_WorkflowIndex.md` — Workflows may reference Principles as steps
- `_ImplementationIndex.md` — Implementations may reference Principles they implement

**Example:**
```markdown
# BEFORE (in _ImplementationIndex.md):
| {repo_name}_FastLanguageModel | [→](...) | ⬜Principle:{repo_name}_LoRA, ⬜Env:{repo_name}_CUDA | ... |

# AFTER (you created LoRA Principle):
| {repo_name}_FastLanguageModel | [→](...) | ✅Principle:{repo_name}_LoRA, ⬜Env:{repo_name}_CUDA | ... |
```

**How to find references:**
- Read each index file
- Search for `⬜Principle:{repo_name}_YourPageName`
- Update all occurrences to `✅Principle:{repo_name}_YourPageName`

## Output Instructions

Write .md files to: `{wiki_dir}/principles/`

**Filename format:** `{repo_name}_Principle_Name.md` (use underscores, descriptive names)

**Examples:**
- `{repo_name}_Low_Rank_Adaptation.md`
- `{repo_name}_Quantization.md`
- `{repo_name}_Flash_Attention.md`

Also UPDATE existing files in: `{wiki_dir}/workflows/`

## Repo Scoping Rule (CRITICAL)

Only create/update pages whose filenames start with `{repo_name}_`.

## ⚠️ File Editing Tip

When updating index files (`_RepoMap.md`, `_PrincipleIndex.md`, `_ImplementationIndex.md`):
- **Use Write tool** (read entire file → modify → write back)
- **Avoid Edit tool** — it often fails on markdown tables

## 📝 Execution Report (REQUIRED)

When finished, write a summary report to `{wiki_dir}/_reports/phase3_synthesis.md`:

```markdown
# Phase 3: Synthesis Report

## Principles Created
| Principle | Implemented By | In Workflows |
|-----------|----------------|--------------|
| [Name] | [impl pages] | [workflow steps] |

## Concept Coverage
- Theoretical concepts documented: X
- Implementations linked: X

## Notes for Enrichment Phase
- [Files with potential environment requirements]
- [Code with heuristics/tribal knowledge]
```
