# Knowledge Search MCP Server

Exposes the knowledge search functionality as an MCP (Model Context Protocol) server for integration with Claude Code, Cursor, and other MCP-compatible AI agents.

## Features

- **Multiple search backends**: Switch between `kg_graph_search` (fast hybrid search) and `kg_llm_navigation` (LLM-guided navigation)
- **Full CRUD operations**: Search, get page, index new pages, edit existing pages
- **Graph enrichment**: Results include connected pages from Neo4j graph

## Available Tools

| Tool | Description | Required Params |
|------|-------------|-----------------|
| `search_knowledge` | Semantic search with filters | `query` |
| `get_wiki_page` | Retrieve page by exact title | `page_title` |
| `kg_index` | Index pages into KG | `wiki_dir` or `page_data` |
| `kg_edit` | Edit existing page | `page_id`, `updates` |
| `list_page_types` | List available page types | none |
| `search_with_context` | Search with additional context | `query` |

## Search Backends

| Backend | Description | Best For |
|---------|-------------|----------|
| `kg_graph_search` | Hybrid vector + graph search using Weaviate embeddings + Neo4j | Fast queries, general search |
| `kg_llm_navigation` | LLM-guided multi-hop navigation | Complex queries needing reasoning |

## Quick Start

### 1. Run MCP Server Directly

```bash
# Use default backend (kg_graph_search)
python -m src.knowledge.wiki_mcps.mcp_server

# Use LLM navigation backend
KG_SEARCH_BACKEND=kg_llm_navigation python -m src.knowledge.wiki_mcps.mcp_server
```

### 2. Configure Claude Code

Copy the template to Claude Code's config directory:

```bash
# Create config directory if needed
mkdir -p ~/.config/claude-code

# Copy and customize the template
cp src/knowledge/wiki_mcps/mcp_config_template.json ~/.config/claude-code/settings.json
```

Or manually add to `~/.config/claude-code/settings.json`:

```json
{
  "mcpServers": {
    "kg-graph-search": {
      "command": "python",
      "args": ["-m", "src.knowledge.wiki_mcps.mcp_server"],
      "cwd": "/home/ubuntu/tinkerer",
      "env": {
        "PYTHONPATH": "/home/ubuntu/tinkerer",
        "KG_SEARCH_BACKEND": "kg_graph_search"
      }
    },
    "kg-llm-navigation": {
      "command": "python",
      "args": ["-m", "src.knowledge.wiki_mcps.mcp_server"],
      "cwd": "/home/ubuntu/tinkerer",
      "env": {
        "PYTHONPATH": "/home/ubuntu/tinkerer",
        "KG_SEARCH_BACKEND": "kg_llm_navigation"
      }
    }
  }
}
```

### 3. Test with Claude Code

```bash
# Verify MCP server is connected
claude -p "List the available tools from the knowledge base"

# Search for knowledge
claude -p "Search for LoRA fine-tuning workflows"
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `KG_SEARCH_BACKEND` | Search backend to use | `kg_graph_search` |
| `OPENAI_API_KEY` | Required for embeddings | - |
| `NEO4J_URI` | Neo4j connection URI | `bolt://localhost:7687` |
| `NEO4J_USER` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | - |
| `WEAVIATE_URL` | Weaviate server URL | `http://localhost:8081` |

## Tool Usage Examples

### Search for Knowledge

```python
# Basic search
search_knowledge(query="How to fine-tune LLM with limited GPU memory?")

# Filtered search
search_knowledge(
    query="fine-tune transformer",
    page_types=["Workflow", "Heuristic"],
    domains=["LLMs", "PEFT"],
    top_k=5
)
```

### Get a Specific Page

```python
get_wiki_page(page_title="QLoRA_Finetuning")
```

### Index New Pages

```python
# Index from directory
kg_index(wiki_dir="/path/to/wikis")

# Index single page
kg_index(page_data={
    "page_title": "My_New_Workflow",
    "page_type": "Workflow",
    "overview": "A workflow for training models...",
    "content": "== Overview ==\nThis workflow...",
    "domains": ["LLMs", "Training"]
})
```

### Edit Existing Page

```python
kg_edit(
    page_id="Workflow/QLoRA_Finetuning",
    updates={
        "overview": "Updated overview text...",
        "domains": ["LLMs", "Fine_Tuning", "Memory_Efficient"]
    }
)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Claude Code / Cursor                      │
└─────────────────────────────┬───────────────────────────────┘
                              │ MCP Protocol (JSON-RPC/stdio)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     MCP Server                               │
│                 (mcp_server.py)                              │
│  ┌───────────┬───────────┬───────────┬──────────────────┐  │
│  │  search   │  get_page │  kg_index │     kg_edit      │  │
│  └─────┬─────┴─────┬─────┴─────┬─────┴────────┬─────────┘  │
│        └───────────┴───────────┴──────────────┘             │
│                            │                                 │
│              KG_SEARCH_BACKEND env var                      │
│                            │                                 │
│        ┌───────────────────┴───────────────────┐            │
│        ▼                                       ▼            │
│  ┌────────────────┐                 ┌────────────────────┐  │
│  │ KGGraphSearch  │                 │KGLLMNavigationSearch│ │
│  │ (fast, hybrid) │                 │ (multi-hop, smart)  │ │
│  └────────────────┘                 └────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
        ┌──────────┐                   ┌──────────┐
        │ Weaviate │                   │  Neo4j   │
        │(vectors) │                   │ (graph)  │
        └──────────┘                   └──────────┘
```

## Troubleshooting

### MCP server not starting

1. Check Python path: `echo $PYTHONPATH`
2. Verify dependencies: `pip install mcp`
3. Check environment variables are set

### Search returning empty results

1. Verify Weaviate is running: `curl http://localhost:8081/v1/.well-known/ready`
2. Verify Neo4j is running: `cypher-shell -u neo4j -p password "MATCH (n) RETURN count(n)"`
3. Check if data is indexed: Run `kg_index` first

### Connection errors

1. Check `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
2. Check `WEAVIATE_URL`
3. Check `OPENAI_API_KEY` for embeddings
