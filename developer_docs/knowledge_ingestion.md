# Knowledge Ingestion & Merge System

Brief overview of the knowledge learning pipeline in `src/knowledge/learners/`.

---

## Architecture

```
Source → Ingestor → WikiPages → Merger → Updated KG
         (Stage 1)              (Stage 2)
```

**Storage:**
- **Neo4j**: Graph index (nodes + edges)
- **Weaviate**: Embedding vectors for semantic search
- **Disk**: `.md` files in `data/wikis/` organized by type

---

## Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `KnowledgePipeline` | `knowledge_learner_pipeline.py` | Main orchestrator, coordinates ingestion + merge |
| `IngestorFactory` | `ingestors/factory.py` | Factory pattern with `@register_ingestor` decorator |
| `RepoIngestor` | `ingestors/repo_ingestor/__init__.py` | 7-phase pipeline for Git repos |
| `KnowledgeMerger` | `knowledge_merger.py` | Merges WikiPages into KG via Claude agent |
| `MergeHandler` | `merge_handlers/*.py` | Type-specific merge logic (Workflow, Principle, etc.) |
| `Source` | `sources.py` | Typed wrappers: `Source.Repo()`, `Source.Paper()` |

---

## RepoIngestor Pipeline (7 Phases)

### Branch 1: Workflow-Based Extraction
| Phase | Name | What It Does |
|-------|------|--------------|
| 0 | `repo_understanding` | Parse AST, generate `_RepoMap.md` |
| 1a | `anchoring` | Find workflows from README → write Workflow pages |
| 1b | `anchoring_context` | Enrich WorkflowIndex with implementation details |
| 2 | `excavation_synthesis` | Trace imports → write Implementation-Principle pairs |
| 3 | `enrichment` | Mine constraints → write Environment/Heuristic pages |
| 4 | `audit` | Validate graph, fix broken links |

### Branch 2: Orphan Mining (uncovered files)

**Why this exists:** Branch 1 extracts knowledge by following workflows (README → code paths). But many useful files (utilities, helpers, configs) aren't referenced by any workflow. Without Branch 2, these "orphan" files would be silently ignored.

**How it works:** A 4-step pipeline that triages all unvisited files, has an agent review ambiguous cases, creates wiki pages for valuable ones, and verifies nothing was missed.

| Step | Name | Type | What It Does |
|------|------|------|--------------|
| 5a | Triage | Code | Deterministic rules classify files: AUTO_KEEP (clearly useful), AUTO_DISCARD (tests, configs, boilerplate), MANUAL_REVIEW (ambiguous) |
| 5b | Review | Agent | LLM evaluates each MANUAL_REVIEW file: "Is this worth documenting?" → approve/reject |
| 5c | Create | Agent | For all approved files (AUTO_KEEP + approved MANUAL_REVIEW), create Implementation/Principle pages |
| 5d | Verify | Code | Sanity check: every approved file should now have a wiki page. Report any gaps. |
| 6 | Orphan Audit | Agent | Final pass: check if any orphan reveals a hidden workflow we missed, flag dead code |

---

## Merger Flow

1. Check if KG exists (query Neo4j for WikiPage nodes)
2. **No index?** → Create all pages as new
3. **Has index?** → For each page:
   - Get type-specific `MergeHandler`
   - Agent searches for related pages via MCP tools
   - Agent decides: MERGE or CREATE
   - Agent executes via `kg_index` or `kg_edit`

---

## Page Types & Merge Semantics

| Type | Match By | Merge Action |
|------|----------|--------------|
| Workflow | Goal/process | Combine steps |
| Principle | Theoretical concept | Merge explanations |
| Implementation | API/function | Merge examples |
| Environment | Tech stack | Merge dependencies |
| Heuristic | Problem domain | Avoid contradictions |

---

## Tests

| Test | Purpose | Run With |
|------|---------|----------|
| `tests/test_repo_ingestor.py` | Single repo ingestion (Bedrock) | `python tests/test_repo_ingestor.py` |
| `tests/run_batch_ingest.py` | Parallel batch ingestion | `python tests/run_batch_ingest.py --workers 5` |
| `tests/test_knowledge_merger.py` | Merge into KG (requires Docker) | `python tests/test_knowledge_merger.py` |

### Environment Variables

```bash
# For Bedrock mode (ingestion)
AWS_BEARER_TOKEN_BEDROCK="..."  # or AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY

# For merger
OPENAI_API_KEY="..."  # For embeddings
NEO4J_URI="bolt://localhost:7687"
NEO4J_PASSWORD="password"
```

---

## Quick Start

```python
from src.knowledge.learners import KnowledgePipeline, Source

# Ingest a repo (skip merge for now)
pipeline = KnowledgePipeline(wiki_dir="data/wikis")
result = pipeline.run(Source.Repo("https://github.com/user/repo"), skip_merge=True)

print(f"Extracted: {result.total_pages_extracted} pages")
```

---

## File Locations

```
src/knowledge/wiki_structure/          # Wiki page & section definitions
├── page_definition.md                 # Page types (Workflow, Principle, etc.)
└── sections_definition.md             # Required sections per page type

src/knowledge/learners/
├── knowledge_learner_pipeline.py   # Main pipeline
├── knowledge_merger.py             # Merger logic
├── sources.py                      # Source types
├── ingestors/
│   ├── base.py                     # Ingestor interface
│   ├── factory.py                  # Factory + registry
│   └── repo_ingestor/
│       ├── __init__.py             # RepoIngestor (7 phases)
│       ├── context_builder.py      # AST parsing, orphan detection
│       ├── utils.py                # Clone/cleanup, wiki structure loading
│       └── prompts/                # Phase prompt templates
└── merge_handlers/
    ├── base.py                     # MergeHandler interface
    ├── workflow_handler.py         # Workflow merge logic
    ├── principle_handler.py        # Principle merge logic
    └── ...                         # Other type handlers
```

