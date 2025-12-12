# Wiki MCP Server

MCP (Model Context Protocol) server that exposes the knowledge search functionality to AI coding agents like Claude Code, Cursor, and other MCP-compatible tools.

## Overview

This module wraps the `src/knowledge/search` backend as an MCP server, enabling seamless integration with AI-powered development environments. MCP is an open standard by Anthropic that allows AI models to interact with external tools using JSON-RPC 2.0.

```
┌─────────────────────────────────────────────────────────┐
│                    AI Coding Agent                       │
│              (Claude Code / Cursor / etc.)              │
│                                                         │
│   ┌─────────────┐                                       │
│   │ MCP Client  │ ◄────── JSON-RPC 2.0 / stdio ───────┐ │
│   └─────────────┘                                      │ │
└─────────────────────────────────────────────────────────┘ │
                                                            │
┌───────────────────────────────────────────────────────────▼─┐
│                    Wiki MCP Server                          │
│                                                             │
│   Tools:                    Resources:                      │
│   - search_knowledge        - knowledge://overview          │
│   - get_wiki_page           - knowledge://page-types        │
│   - list_page_types                                         │
│   - search_with_context                                     │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              KGGraphSearch Backend                   │   │
│   │         (Weaviate + Neo4j + LLM Reranker)           │   │
│   └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### 1. Install MCP Package

```bash
pip install mcp
```

### 2. Ensure Backend Services are Running

The MCP server requires the knowledge search backend services:

```bash
# Start Weaviate (vector database)
docker run -d --name weaviate \
    -p 8081:8080 -p 50051:50051 \
    -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
    -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \
    semitechnologies/weaviate:latest

# Start Neo4j (graph database)
docker run -d --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password123 \
    neo4j:latest
```

### 3. Set Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password123"
export WEAVIATE_URL="http://localhost:8081"
```

## Usage

### Running the MCP Server

```bash
# From project root
python -m src.knowledge.wiki_mcps.mcp_server

# Or with uv
uv run src/knowledge/wiki_mcps/mcp_server.py
```

### Configuration for Cursor

Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "knowledge-search": {
      "command": "python",
      "args": ["-m", "src.knowledge.wiki_mcps.mcp_server"],
      "cwd": "/path/to/praxium",
      "env": {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "${NEO4J_PASSWORD}",
        "WEAVIATE_URL": "http://localhost:8081"
      }
    }
  }
}
```

### Configuration for Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "knowledge-search": {
      "command": "python",
      "args": ["-m", "src.knowledge.wiki_mcps.mcp_server"],
      "cwd": "/path/to/praxium"
    }
  }
}
```

## Available Tools

### search_knowledge

Search the ML/AI knowledge base with semantic search and LLM reranking.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `query` | string | Yes | Natural language search query |
| `top_k` | integer | No | Number of results (default: 5, max: 20) |
| `page_types` | array | No | Filter: Workflow, Principle, Implementation, Environment, Heuristic |
| `domains` | array | No | Filter: LLMs, Deep_Learning, NLP, PEFT, etc. |
| `min_score` | number | No | Minimum relevance score (0.0-1.0) |

**Example:**
```json
{
  "query": "How to fine-tune LLM with limited GPU memory",
  "top_k": 5,
  "page_types": ["Workflow", "Heuristic"],
  "domains": ["LLMs", "PEFT"]
}
```

### get_wiki_page

Retrieve a specific wiki page by exact title.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `page_title` | string | Yes | Exact page title |

**Example:**
```json
{
  "page_title": "QLoRA_Finetuning"
}
```

### list_page_types

List all available page types with descriptions.

**Parameters:** None

### search_with_context

Search with additional context for better relevance.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `query` | string | Yes | Search query |
| `context` | string | No | Additional context (experiment setup, errors, etc.) |
| `top_k` | integer | No | Number of results |
| `page_types` | array | No | Filter by page types |

## Available Resources

| URI | Description |
|-----|-------------|
| `knowledge://overview` | Overview of the knowledge base structure |
| `knowledge://page-types` | Detailed reference of all page types |

## Testing

### Test with MCP Inspector

```bash
npx @modelcontextprotocol/inspector python -m src.knowledge.wiki_mcps.mcp_server
```

### Test Programmatically

```python
from src.knowledge.wiki_mcps import create_mcp_server

# Create server instance
server = create_mcp_server()

# The server exposes:
# - list_tools() -> List available tools
# - call_tool(name, arguments) -> Execute a tool
# - list_resources() -> List available resources
# - read_resource(uri) -> Read a resource
```

## Architecture

### MCP Primitives Used

1. **Tools**: Functions the AI can call
   - `search_knowledge` - Semantic search
   - `get_wiki_page` - Direct page retrieval
   - `list_page_types` - Type reference
   - `search_with_context` - Contextual search

2. **Resources**: Data the AI can read
   - Overview documentation
   - Page types reference

### Search Pipeline

```
Query ──► Embedding ──► Weaviate Search ──► LLM Reranker ──► Graph Enrichment ──► Results
              │              (top 2K)           (optional)        (Neo4j)           │
              │                                                                     │
              └─────────────────── OpenAI text-embedding-3-large ───────────────────┘
```

## File Structure

```
src/knowledge/wiki_mcps/
├── __init__.py        # Module exports
├── mcp_server.py      # MCP server implementation
└── README.md          # This file
```

## Troubleshooting

### MCP package not found

```bash
pip install mcp
```

### Connection errors to Weaviate/Neo4j

Ensure services are running:
```bash
docker ps  # Check containers
docker logs weaviate  # Check Weaviate logs
docker logs neo4j  # Check Neo4j logs
```

### Empty search results

1. Verify the knowledge base is indexed:
   ```python
   from src.knowledge.search import KnowledgeSearchFactory, KGIndexInput
   search = KnowledgeSearchFactory.create("kg_graph_search")
   search.index(KGIndexInput(wiki_dir="data/wikis"))
   ```

2. Check environment variables are set correctly

### Server not appearing in Cursor/Claude

1. Verify the config file path is correct
2. Restart the AI application
3. Check logs for connection errors

## Related Documentation

- [Knowledge Search Module](../search/README.md)
- [MCP Specification](https://modelcontextprotocol.io/specification)
- [Wiki Structure Definition](../wiki_structure/summary.md)

