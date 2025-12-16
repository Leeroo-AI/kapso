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
│   Git Repo ──► Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──►              │
│                  │           │           │           │                       │
│               RepoMap    Workflows   Implement.  Principles                  │
│                                                                              │
│            ──► Phase 4 ──► Phase 5 ──► Phase 6 (multi-step) ──► Phase 7     │
│                  │           │           │                         │         │
│              Env/Heur     Audit    6a→6b→6c→6d              Orphan Audit    │
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

### Phase 2: Excavation

**Purpose:** Trace imports from Workflows to find the actual implementation code.

**How it works:**
1. Reads Workflow pages to identify which APIs are used
2. Locates source files using the Repository Map
3. Creates **Implementation pages** with:
   - Code Reference (GitHub URL with line numbers)
   - Full signature with types
   - I/O Contract (inputs/outputs as tables)
   - Usage Examples (runnable code)

**Output:** `implementations/{repo_name}_ClassName.md`

---

### Phase 3: Synthesis

**Purpose:** Extract theoretical concepts and design patterns from implementations.

**How it works:**
1. Analyzes Implementation pages for underlying concepts
2. Creates **Principle pages** explaining the "why" behind the code
3. Links Principles to their Implementation(s)

**Examples:**
- "LoRA: Low-Rank Adaptation"
- "Flash Attention"
- "Gradient Checkpointing"

**Output:** `principles/{repo_name}_PrincipleName.md`

---

### Phase 4: Enrichment

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

### Phase 5: Audit

**Purpose:** Validate the knowledge graph integrity.

**How it works:**
1. Checks all internal wiki links resolve to real pages
2. Verifies mandatory relationships exist
3. Cross-references page indexes for consistency
4. Flags issues for repair

**Output:** Validation report + fixes to broken links

---

### Phase 6: Orphan Mining (Multi-Step Pipeline)

**Purpose:** Find code that wasn't captured through workflow-based analysis.

Phase 6 is a **4-step pipeline** that combines deterministic code-based filtering with agent evaluation:

#### Step 6a: Triage (Code-Based)

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

#### Step 6b: Review (Agent)

**Executor:** Claude Code agent

**How it works:**
1. Reads `_orphan_candidates.md`
2. For each file in MANUAL_REVIEW section:
   - Reads the source file
   - Evaluates: public API? user-facing? distinct algorithm?
   - Writes decision: `✅ APPROVED` or `❌ REJECTED` with reasoning

**Output:** Updated `_orphan_candidates.md` with decisions filled in

---

#### Step 6c: Create (Agent)

**Executor:** Claude Code agent

**How it works:**
1. Reads approved files (AUTO_KEEP + APPROVED from MANUAL_REVIEW)
2. For each approved file:
   - Creates Implementation page
   - Creates Principle page if needed (with polymorphism check)
   - Updates Status column to `✅ DONE` (checkpoint)
   - Updates RepoMap Coverage column
   - Updates page indexes

**Output:** Implementation/Principle pages, updated indexes

---

#### Step 6d: Verify (Code-Based)

**Executor:** Python code (deterministic, no agent)

**How it works:**
1. Checks all AUTO_KEEP files have `DONE` status
2. Checks all MANUAL_REVIEW files have decisions (not PENDING)
3. Checks all approved files have wiki pages

**Output:** Verification report (pass/fail with errors)

---

### Phase 7: Orphan Audit

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

Generated by Step 6a, tracks orphan file processing:

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
- `phase2_excavation.md`
- `phase3_synthesis.md`
- `phase4_enrichment.md`
- `phase5_audit.md`
- `phase6b_orphan_review.md`
- `phase6c_orphan_create.md`
- `phase6d_orphan_verify.md` (if errors)

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
Phase 2 ──────────────────────────────────────────────────────────────►
         │ Reads Workflows, traces imports → Creates Implementations
         │ Updates Coverage column in RepoMap
         ▼
Phase 3 ──────────────────────────────────────────────────────────────►
         │ Reads Implementations → Creates Principles
         │ Links Principles ↔ Implementations
         ▼
Phase 4 ──────────────────────────────────────────────────────────────►
         │ Scans code for constraints → Creates Environments + Heuristics
         │ Links to Implementations
         ▼
Phase 5 ──────────────────────────────────────────────────────────────►
         │ Validates all links, fixes broken references
         ▼
Phase 6a (CODE) ──────────────────────────────────────────────────────►
         │ Reads Coverage column → Applies D1-D5, K1-K3 rules
         │ Creates _orphan_candidates.md
         ▼
Phase 6b (AGENT) ─────────────────────────────────────────────────────►
         │ Evaluates MANUAL_REVIEW files → Writes decisions
         ▼
Phase 6c (AGENT) ─────────────────────────────────────────────────────►
         │ Creates pages for approved files → Checkpoints progress
         │ Updates RepoMap Coverage + page indexes
         ▼
Phase 6d (CODE) ──────────────────────────────────────────────────────►
         │ Verifies all approved files have pages
         ▼
Phase 7 ──────────────────────────────────────────────────────────────►
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

7. **Deterministic Orphan Triage:** Phase 6 uses code-based filtering (D1-D5, K1-K3 rules) for reliable, reproducible results. Agent judgment is limited to borderline cases (MANUAL_REVIEW).

8. **Checkpointing in Phase 6c:** Progress is saved after each page creation, allowing resumption if interrupted.

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Phase 0 stops early | Agent didn't explore all files | Verification loop retries with feedback |
| Missing Coverage updates | Agent forgot to update index | Prompts explicitly require index sync |
| Broken wiki links | Page renamed or missing | Audit phase detects and fixes |
| File paths in Metadata | Confused high-level vs low-level refs | Prompts now have ❌/✅ examples |
| Orphan files missed | Subjective filtering | Step 6a now uses deterministic rules |
| Orphan progress lost | No checkpointing | Step 6c checkpoints each page |

---

## Summary

The RepoIngestor transforms raw code into structured knowledge through multiple phases:

**Branch 1: Workflow-Based Extraction**
1. **Understand** the repository structure (Phase 0)
2. **Anchor** on workflows/use cases (Phase 1)
3. **Excavate** implementation details (Phase 2)
4. **Synthesize** underlying principles (Phase 3)
5. **Enrich** with environment/heuristics (Phase 4)
6. **Audit** for integrity (Phase 5)

**Branch 2: Orphan Mining (Multi-Step)**
7. **Triage** orphan candidates deterministically (Step 6a)
8. **Review** borderline files with agent judgment (Step 6b)
9. **Create** wiki pages for approved files (Step 6c)
10. **Verify** all approved files have pages (Step 6d)
11. **Audit** orphans for hidden workflows (Phase 7)

The output is a complete knowledge graph that other agents can query to understand, use, and modify the codebase.
