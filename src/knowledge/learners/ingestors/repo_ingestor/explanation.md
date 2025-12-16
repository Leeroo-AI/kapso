# Repository Ingestor: Knowledge Extraction Pipeline

## Overview

The **RepoIngestor** is a prompt-driven, multi-phase knowledge extraction pipeline that transforms Git repositories into structured wiki pages. It uses a coding agent (Claude Code) to read source code and produce a knowledge graph with five page types: Workflows, Implementations, Principles, Environments, and Heuristics.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Repository Ingestor                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Git Repo ──► Phase 0 ──► Phase 1 ──► Phase 2 ──────────────────►          │
│                  │           │           │                                   │
│               RepoMap    Workflows   Impl+Principle PAIRS                   │
│                                      (merged excavation+synthesis)           │
│                                                                              │
│            ──► Phase 3 ──► Phase 4 ──► Phase 5 (multi-step) ──► Phase 6     │
│                  │           │           │                         │         │
│              Env/Heur     Audit    5a→5b→5c→5d              Orphan Audit    │
│                                    (Triage→Review→Create→Verify)            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## The Phases

### Phase 0: Repository Understanding

**Purpose:** Build a structural map of the entire repository before any knowledge extraction.

**How it works:**
1. AST parser scans all Python files extracting classes, functions, imports
2. Creates a compact **index file** (`_RepoMap_{repo_name}.md`) with:
   - Status (explored/pending)
   - Purpose (brief description)
   - Coverage (which wiki pages cover this file)
3. Creates **detail files** (`_files/*.md`) for each source file
4. Agent fills in "Understanding" sections for every file

**Output:**
- `_RepoMap_{repo_name}.md` — Central index with all files
- `_files/*.md` — Per-file detail pages with AST info + natural language understanding

**Why it matters:** Subsequent phases can quickly navigate the codebase without re-reading source files.

---

### Phase 1: Anchoring

**Purpose:** Identify and document the "Golden Paths" — primary use cases and workflows.

**How it works:**
1. Reads Phase 0 report and the Repository Map
2. Finds example files, tutorials, and entry points
3. Creates **Workflow pages** documenting step-by-step processes

**Example workflows:**
- "QLoRA Fine-tuning"
- "Model Export to GGUF"
- "Vision-Language Training"

**Output:** `workflows/{repo_name}_WorkflowName.md`

---

### Phase 2: Excavation + Synthesis (Merged)

**Purpose:** Trace imports from Workflows to source code, then create **Implementation-Principle PAIRS together** to keep concepts tightly connected to their implementations.

**Why merged:** Previously separate phases caused disconnect — Principles were written without fresh context of the implementation details. Now both pages are written atomically while the agent has full understanding.

**How it works:**
For EACH API/function traced from workflows:
1. Read the source code
2. Write **Implementation page** (HOW it works):
   - Code Reference (GitHub URL with line numbers)
   - Full signature with types
   - I/O Contract (inputs/outputs as tables)
   - Usage Examples (runnable code)
3. Write **Principle page** (WHY/WHAT concept):
   - Theoretical explanation
   - Mathematical basis
   - When to use
4. Link them bidirectionally
5. Move to next API

**Multi-Implementation Principles:** When a Principle spans multiple implementations (e.g., "LoRA Merging" → save_pretrained_merged + save_pretrained_gguf), an **Implementation Mapping** table is included that explicitly shows which part of the concept each implementation handles:

```
| Concept Component | Implementation | What It Does |
|-------------------|----------------|--------------|
| LoRA weight fusion | save_pretrained_merged | Merges adapters, saves HuggingFace format |
| LoRA merge + GGUF | save_pretrained_gguf | Merges adapters, converts to GGUF |
```

**Output:**
- `implementations/{repo_name}_ClassName.md` — Implementation pages
- `principles/{repo_name}_ConceptName.md` — Principle pages (linked)

---

### Phase 3: Enrichment

**Purpose:** Extract environment requirements and tribal knowledge (heuristics).

**How it works:**
1. Scans for version checks, device requirements, dependency constraints
2. Mines comments/docstrings for tips, warnings, magic numbers
3. Creates **Environment pages** (e.g., "PyTorch_CUDA_11_8")
4. Creates **Heuristic pages** (e.g., "Batch_Size_Memory_Tradeoff")

**Output:**
- `environments/{repo_name}_EnvName.md`
- `heuristics/{repo_name}_HeuristicName.md`

---

### Phase 4: Audit

**Purpose:** Validate the knowledge graph integrity.

**How it works:**
1. Checks all internal wiki links resolve to real pages
2. Verifies mandatory relationships exist
3. Cross-references page indexes for consistency
4. Flags issues for repair

**Output:** Validation report + fixes to broken links

---

### Phase 5: Orphan Mining (Multi-Step Pipeline)

**Purpose:** Find code that wasn't captured through workflow-based analysis.

Phase 5 is a **4-step pipeline** that combines deterministic code-based filtering with agent evaluation:

#### Step 5a: Triage (Code-Based)

**Executor:** Python code (deterministic, no agent)

**How it works:**
1. Reads the Repository Map's **Coverage column**
2. Files with `—` coverage are orphan candidates
3. Applies deterministic filter rules:

**AUTO_DISCARD Rules (skip these files):**
| Rule | Condition |
|------|-----------|
| D1 | ≤20 lines |
| D2 | `__init__.py` with <100 lines |
| D3 | Test files (`/tests/`, `test_*.py`, `*_test.py`) |
| D4 | Benchmark files (`/benchmark/`) |
| D5 | Scripts directory (`scripts/`) |

**AUTO_KEEP Rules (must document):**
| Rule | Condition |
|------|-----------|
| K1 | ≥300 lines |
| K2 | Kernel files (`/kernels/`) with ≥100 lines |
| K3 | Model files (`/models/`) with ≥200 lines |

**MANUAL_REVIEW:** Everything else (agent decides)

**Output:** `_orphan_candidates.md` with three sections

---

#### Step 5b: Review (Agent)

**Executor:** Claude Code agent

**How it works:**
1. Reads `_orphan_candidates.md`
2. For each file in MANUAL_REVIEW section:
   - Reads the source file
   - Evaluates: public API? user-facing? distinct algorithm?
   - Writes decision: `✅ APPROVED` or `❌ REJECTED` with reasoning

**Output:** Updated `_orphan_candidates.md` with decisions filled in

---

#### Step 5c: Create (Agent)

**Executor:** Claude Code agent

**How it works:**
1. Reads approved files (AUTO_KEEP + APPROVED from MANUAL_REVIEW)
2. For each approved file:
   - Creates Implementation-Principle pair (using same merged approach as Phase 2)
   - Updates Status column to `✅ DONE` (checkpoint)
   - Updates RepoMap Coverage column
   - Updates page indexes

**Output:** Implementation/Principle pages, updated indexes

---

#### Step 5d: Verify (Code-Based)

**Executor:** Python code (deterministic, no agent)

**How it works:**
1. Checks all AUTO_KEEP files have `DONE` status
2. Checks all MANUAL_REVIEW files have decisions (not PENDING)
3. Checks all approved files have wiki pages

**Output:** Verification report (pass/fail with errors)

---

### Phase 6: Orphan Audit

**Purpose:** Quality control for orphan nodes.

**How it works:**
1. Checks if orphans have hidden workflows (should create Workflow page)
2. Validates orphan pages follow schema
3. Ensures page index consistency

**Output:** Final validated wiki pages

---

## Wiki Page Types

| Page Type | Purpose | Example |
|-----------|---------|---------|
| **Workflow** | Step-by-step process | "QLoRA Fine-tuning Pipeline" |
| **Implementation** | Concrete code entity | "FastLanguageModel class" |
| **Principle** | Theoretical concept | "Low-Rank Adaptation (LoRA)" |
| **Environment** | Runtime requirements | "PyTorch 2.0 + CUDA 11.8" |
| **Heuristic** | Tribal knowledge/tips | "Use gradient checkpointing for 7B+ models" |

---

## Key Artifacts

### Repository Map (`_RepoMap_{repo_name}.md`)

A compact index tracking all source files:

```
| Status | File | Purpose | Coverage | Details |
|--------|------|---------|----------|---------|
| ✅ | loader.py | Main model loader | Impl: FastLanguageModel | [→](_files/...) |
| ⬜ | utils.py | Helper utilities | — | [→](_files/...) |
```

### Orphan Candidates (`_orphan_candidates.md`)

Generated by Step 5a, tracks orphan file processing:

```
## AUTO_KEEP (Must Document)
| # | File | Lines | Rule | Status |
|---|------|-------|------|--------|
| 1 | path/file.py | 500 | K1: ≥300 lines | ⬜ PENDING |

## MANUAL_REVIEW (Agent Evaluates)
| # | File | Lines | Purpose | Decision | Reasoning |
|---|------|-------|---------|----------|-----------|
| 1 | path/util.py | 150 | Utility | ⬜ PENDING | |
```

### Page Indexes

Separate indexes for each page type:
- `_WorkflowIndex.md` — Workflows with step connections
- `_ImplementationIndex.md` — Implementations with source locations
- `_PrincipleIndex.md` — Principles with linked implementations
- `_EnvironmentIndex.md` — Environment requirements
- `_HeuristicIndex.md` — Tips and tricks

### Phase Reports (`_reports/`)

Each phase writes a summary report:
- `phase0_repo_understanding.md`
- `phase1_anchoring.md`
- `phase2_excavation_synthesis.md`
- `phase3_enrichment.md`
- `phase4_audit.md`
- `phase5b_orphan_review.md`
- `phase5c_orphan_create.md`
- `phase5d_orphan_verify.md` (if errors)

Reports contain: summary, statistics, key discoveries, and notes for the next phase.

---

## Data Flow Between Phases

```
Phase 0 ──────────────────────────────────────────────────────────────►
         │ Creates RepoMap + file details
         ▼
Phase 1 ──────────────────────────────────────────────────────────────►
         │ Uses RepoMap to find examples → Creates Workflows
         ▼
Phase 2 (MERGED Excavation+Synthesis) ────────────────────────────────►
         │ Reads Workflows, traces imports
         │ For EACH API: Creates Implementation + Principle PAIR together
         │ Multi-impl Principles include Implementation Mapping table
         │ Updates Coverage column in RepoMap
         ▼
Phase 3 ──────────────────────────────────────────────────────────────►
         │ Scans code for constraints → Creates Environments + Heuristics
         │ Links to Implementations
         ▼
Phase 4 ──────────────────────────────────────────────────────────────►
         │ Validates all links, fixes broken references
         ▼
Phase 5a (CODE) ──────────────────────────────────────────────────────►
         │ Reads Coverage column → Applies D1-D5, K1-K3 rules
         │ Creates _orphan_candidates.md
         ▼
Phase 5b (AGENT) ─────────────────────────────────────────────────────►
         │ Evaluates MANUAL_REVIEW files → Writes decisions
         ▼
Phase 5c (AGENT) ─────────────────────────────────────────────────────►
         │ Creates Implementation-Principle pairs for approved files
         │ Uses same merged approach as Phase 2
         │ Updates RepoMap Coverage + page indexes
         ▼
Phase 5d (CODE) ──────────────────────────────────────────────────────►
         │ Verifies all approved files have pages
         ▼
Phase 6 ──────────────────────────────────────────────────────────────►
         │ Final validation, hidden workflow check
         ▼
       DONE: Complete Knowledge Graph
```

---

## Implementation Page Structure

Implementation pages are the most detailed. Here's their structure:

```
┌─────────────────────────────────────────┐
│ Metadata Block (top-right wikitable)    │
│ - Sources: Repo URL, Docs, Papers       │
│ - Domains: NLP, Training, etc.          │
│ - Last Updated                          │
├─────────────────────────────────────────┤
│ == Overview ==                          │
│ One-sentence description                │
├─────────────────────────────────────────┤
│ == Code Reference ==                    │
│ - Source Location (GitHub URL + lines)  │
│ - Full Signature                        │
│ - Import Statement                      │
├─────────────────────────────────────────┤
│ == I/O Contract ==                      │
│ - Inputs table (name, type, required)   │
│ - Outputs table (name, type, desc)      │
├─────────────────────────────────────────┤
│ == Usage Examples ==                    │
│ Complete, runnable code snippets        │
├─────────────────────────────────────────┤
│ == Related Pages ==                     │
│ - Links to Environments                 │
│ - Links to Heuristics                   │
└─────────────────────────────────────────┘
```

---

## Running the Ingestor

```python
from src.knowledge.learners.ingestors.repo_ingestor import RepoIngestor

ingestor = RepoIngestor(
    params={
        "wiki_dir": "./output_wiki",
        "timeout": 3600,  # 1 hour per phase
    }
)

# Run all phases
pages = ingestor.ingest({"url": "https://github.com/unslothai/unsloth"})
```

---

## Key Design Decisions

1. **Phase 0 First:** Understanding the repo structure before extraction prevents redundant file reading in later phases.

2. **Coverage Tracking:** The RepoMap's Coverage column prevents duplicate work and enables orphan detection.

3. **Phase Reports:** Each phase writes a summary for the next phase, providing continuity and context.

4. **Separate Page Indexes:** Makes cross-reference validation fast and enables integrity checking.

5. **Code Reference Format:** GitHub URLs with line numbers make pages actionable—agents can locate and modify code.

6. **Usage Examples Required:** Implementation pages must include runnable code to be useful for downstream agents.

7. **Deterministic Orphan Triage:** Phase 5 uses code-based filtering (D1-D5, K1-K3 rules) for reliable, reproducible results. Agent judgment is limited to borderline cases (MANUAL_REVIEW).

8. **Checkpointing in Phase 5c:** Progress is saved after each page creation, allowing resumption if interrupted.

9. **Merged Excavation+Synthesis:** Implementation and Principle pages are written together for each API, keeping concepts tightly connected. Multi-implementation Principles include explicit mapping tables showing which implementation handles which part of the concept.

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Phase 0 stops early | Agent didn't explore all files | Verification loop retries with feedback |
| Missing Coverage updates | Agent forgot to update index | Prompts explicitly require index sync |
| Broken wiki links | Page renamed or missing | Audit phase detects and fixes |
| File paths in Metadata | Confused high-level vs low-level refs | Prompts now have ❌/✅ examples |
| Orphan files missed | Subjective filtering | Step 5a now uses deterministic rules |
| Orphan progress lost | No checkpointing | Step 5c checkpoints each page |
| Principle disconnected from Impl | Separate phases | Merged Phase 2 writes pairs together |

---

## Summary

The RepoIngestor transforms raw code into structured knowledge through multiple phases:

**Branch 1: Workflow-Based Extraction**
1. **Understand** the repository structure (Phase 0)
2. **Anchor** on workflows/use cases (Phase 1)
3. **Excavate+Synthesize** implementation-principle pairs together (Phase 2)
4. **Enrich** with environment/heuristics (Phase 3)
5. **Audit** for integrity (Phase 4)

**Branch 2: Orphan Mining (Multi-Step)**
6. **Triage** orphan candidates deterministically (Step 5a)
7. **Review** borderline files with agent judgment (Step 5b)
8. **Create** wiki pages for approved files (Step 5c)
9. **Verify** all approved files have pages (Step 5d)
10. **Audit** orphans for hidden workflows (Phase 6)

The output is a complete knowledge graph that other agents can query to understand, use, and modify the codebase.
