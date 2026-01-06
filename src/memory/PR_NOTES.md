# Cognitive Memory System - PR Notes

## Overview

This PR adds a cognitive memory system that provides:
- **Workflow-guided context** from Knowledge Graph
- **Graph traversal** for curated knowledge paths
- **Episodic memory** for learning from past experiments
- **LLM-governed decisions** for retry/pivot/complete

## Key Features

### 1. Two-Stage Retrieval
- **Stage 1 (Planning)**: Workflow → Steps → Principles → Heuristics
- **Stage 2 (Implementation)**: Graph traversal to get Implementation pages with code snippets

### 2. Graph Traversal
Uses Neo4j relationships instead of semantic search when workflow is found:
```
Workflow ──STEP──> Principle ──IMPLEMENTED_BY──> Implementation
                       │                              │
                       └──USES_HEURISTIC──> Heuristic └──REQUIRES_ENV──> Environment
```

### 3. Config-Based Switching
Change `context_manager.type` in any mode in `src/config.yaml`:
```yaml
# Switch from legacy to cognitive (one line change)
context_manager:
  type: "cognitive"  # instead of "token_efficient"

# Also enable KG for workflow retrieval
knowledge_search:
  type: "kg_graph_search"
  enabled: true
```

## Files Changed

### Core Memory System (`src/memory/`)
- `cognitive_controller.py` - Main controller with LLM decisions
- `knowledge_retriever.py` - Tiered retrieval with graph traversal
- `context.py` - Unified context rendering
- `episodic.py` - Vector-based insight storage
- `episodic_retriever.py` - LLM-governed episodic retrieval
- `insight_extractor.py` - LLM-based insight generalization
- `decisions.py` - LLM decision making

### Context Manager
- `src/execution/context_manager/cognitive_context_manager.py` - Integration with Expert

### Tests
- `tests/test_expert_full_e2e.py` - Full E2E test with real KG
- `tests/test_cognitive_multi_iteration.py` - Unit tests for components

## Running Tests

```bash
# Full E2E test (requires KG running)
PYTHONPATH=. python tests/test_expert_full_e2e.py

# Unit tests
PYTHONPATH=. python tests/test_cognitive_multi_iteration.py
```

## How to Check Logs

1. Log file: `logs/expert_e2e_YYYYMMDD_HHMMSS.log`
2. Key markers to search:
   - `🎯 GOAL INITIALIZATION` - Goal parsed, workflow retrieved
   - `📋 PREPARING BRIEFING` - Context prepared for agent
   - `⚙️ PROCESSING EXPERIMENT` - Result evaluated
   - `🧠 LLM Decision` - Action decided

## Impact on MLE/ALE

**NONE** - This is a new context manager (`cognitive`) that runs alongside existing:
- `default` - Basic context
- `kg_enriched` - KG-based context without workflow tracking

Switching modes requires config change only.

---

## Heuristic Backlink Handling

Per the KG structure (`src/knowledge/wiki_structure/page_connections.md`), heuristics are
**leaf nodes** that receive edges from Workflow, Principle, and Implementation pages.

However, Heuristic pages can declare "backlinks" to document which pages use them:
```markdown
# In Heuristic page (backlink syntax)
[[uses_heuristic::Principle:some_principle]]
```

The indexer now **automatically reverses** these backlinks during indexing:
- Input: Heuristic page declares `[[uses_heuristic::Principle:X]]`  
- Output: Creates edge `Principle → USES_HEURISTIC → Heuristic` (correct direction!)

This is handled in `src/knowledge/search/kg_graph_search.py` in `_create_neo4j_edges()`.

---

## Retrieval Algorithm Summary

```
1. FIND WORKFLOW (Semantic Search)
   - Search Weaviate for Workflow pages matching goal
   - LLM reranker filters results
   - Threshold: 0.7 confidence

2. GRAPH TRAVERSAL (Neo4j)
   - Workflow → STEP → Principle (ordered steps)
   - Principle → IMPLEMENTED_BY → Implementation (code)
   - Principle → USES_HEURISTIC → Heuristic (tips)
   - Implementation → REQUIRES_ENV → Environment (deps)

3. CONTENT FETCH (Weaviate)
   - For each node ID from Neo4j, fetch overview/content from Weaviate
   - Extract code snippets from <syntaxhighlight> tags

4. FALLBACK (Semantic Search)
   - If no workflow match: TIER 2 synthesizes plan from Principles
   - If error during execution: TIER 3 retrieves error-specific heuristics
```

