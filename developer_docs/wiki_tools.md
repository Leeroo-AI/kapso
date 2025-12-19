# Wiki Tools & MCP Layer

Overview of wiki search/edit tools in `src/knowledge/search/` and MCP wrappers in `src/knowledge/wiki_mcps/`.

---

## Architecture

```
Claude Code / AI Agent
         │
         ▼
┌─────────────────────────────────────────────────┐
│  MCP Server (wiki_mcps/mcp_server.py)           │
│  Exposes tools via JSON-RPC 2.0 over stdio      │
│                                                 │
│  Tools: search_knowledge, get_wiki_page,        │
│         kg_index, kg_edit, list_page_types      │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  KGGraphSearch (search/kg_graph_search.py)      │
│  Hybrid search: Weaviate + Neo4j + LLM rerank   │
└─────────────────────────────────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│Weaviate│ │ Neo4j  │
│Vectors │ │ Graph  │
└────────┘ └────────┘
```

---

## Core Search Backend

**Location:** `src/knowledge/search/kg_graph_search.py`

### Data Structures (`base.py`)

| Class | Purpose |
|-------|---------|
| `WikiPage` | Parsed wiki page with id, title, type, overview, content, domains, links |
| `KGIndexInput` | Input for indexing: `wiki_dir` path or `pages` list |
| `KGEditInput` | Input for editing: `page_id` + fields to update |
| `KGSearchFilters` | Filters: `top_k`, `min_score`, `page_types`, `domains` |
| `KGOutput` | Search results with ranked `KGResultItem` list |

### KGGraphSearch Operations

| Method | Description |
|--------|-------------|
| `index(KGIndexInput)` | Parse wiki files → index to Weaviate + Neo4j |
| `search(query, filters)` | Semantic search + LLM rerank + graph enrichment |
| `get_page(page_title)` | Direct lookup by exact title |
| `edit(KGEditInput)` | Update page across all layers (files, cache, Weaviate, Neo4j) |

### Search Flow

1. **Embed query** → OpenAI `text-embedding-3-large`
2. **Weaviate vector search** → top-k candidates
3. **LLM reranking** → Claude reorders by relevance
4. **Neo4j enrichment** → add connected pages from graph
5. **Return** `KGOutput` with scored results

### Edit Flow (4-layer sync)

1. **Source file** → Update `.md` file on disk
2. **Persist cache** → Update `data/indexes/wikis.json`
3. **Weaviate** → Re-embed if overview changed
4. **Neo4j** → Rebuild edges if links changed

---

## MCP Server

**Location:** `src/knowledge/wiki_mcps/mcp_server.py`

Exposes wiki tools via [MCP (Model Context Protocol)](https://github.com/anthropics/mcp) for AI agents.

### Available Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `search_knowledge` | Semantic search with filters | `query`, `top_k`, `page_types`, `domains` |
| `get_wiki_page` | Get full page by exact title | `page_title` |
| `kg_index` | Index pages from dir or single page | `wiki_dir` or `page_data` |
| `kg_edit` | Edit existing page | `page_id`, `updates` |
| `list_page_types` | List page types with descriptions | — |
| `search_with_context` | Search with experiment context | `query`, `context` |

### Running the MCP Server

```bash
# Direct (stdio transport)
python -m src.knowledge.wiki_mcps.mcp_server

# Use different backend
KG_SEARCH_BACKEND=kg_llm_navigation python -m src.knowledge.wiki_mcps.mcp_server
```

### MCP Config for Claude Code

```json
{
  "mcpServers": {
    "kg-graph-search": {
      "command": "python",
      "args": ["-m", "src.knowledge.wiki_mcps.mcp_server"],
      "cwd": "/path/to/praxium",
      "env": {
        "PYTHONPATH": "/path/to/praxium",
        "KG_SEARCH_BACKEND": "kg_graph_search"
      }
    }
  }
}
```

---

## Test: MCP Integration

**Location:** `tests/test_mcp_integration.py`

### What It Tests

1. MCP server starts correctly
2. Claude Code can connect to it
3. Knowledge search tools work end-to-end

### How It Works

1. Creates temporary MCP config JSON
2. Runs `claude` CLI with `--mcp-config` pointing to the server
3. Sends a test prompt that uses `search_knowledge` tool
4. Verifies Claude completes successfully

### Running the Test

```bash
# Requires ANTHROPIC_API_KEY
python tests/test_mcp_integration.py
```

### Test Prompt Used

```
Build a llama 3 post-training with GRPO on a user preference dataset.
Search the knowledge base for relevant workflows and best practices.
```

---

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `OPENAI_API_KEY` | Embeddings | Required |
| `NEO4J_URI` | Graph database | `bolt://localhost:7687` |
| `NEO4J_USER` | Neo4j auth | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j auth | `password` |
| `WEAVIATE_URL` | Vector database | `http://localhost:8081` |
| `KG_SEARCH_BACKEND` | Backend type | `kg_graph_search` |

---

## File Locations

```
src/knowledge/search/
├── base.py                 # Data structures (WikiPage, KGIndexInput, etc.)
├── factory.py              # KnowledgeSearchFactory
├── kg_graph_search.py      # Main hybrid search implementation
└── kg_llm_navigation.py    # LLM-guided multi-hop navigation (alternative)

src/knowledge/wiki_mcps/
├── mcp_server.py           # MCP server with tool handlers
├── mcp_config_template.json # Example config for Claude Code
└── README.md               # MCP setup guide
```

---

## Quick Usage Examples

### Python API

```python
from src.knowledge.search.factory import KnowledgeSearchFactory
from src.knowledge.search.base import KGSearchFilters, KGIndexInput

# Create search backend
search = KnowledgeSearchFactory.create("kg_graph_search")

# Index pages
search.index(KGIndexInput(wiki_dir="data/wikis"))

# Search
results = search.search(
    query="How to fine-tune LLM with limited GPU?",
    filters=KGSearchFilters(top_k=5, page_types=["Workflow", "Heuristic"])
)

for item in results:
    print(f"{item.page_title} ({item.score:.2f})")
```

### Via MCP (Claude Code)

```
Use search_knowledge to find workflows for LoRA fine-tuning.
Then use get_wiki_page to read the most relevant one.
```

