# Knowledge Search

Semantic search over wiki pages using embeddings (Weaviate), LLM reranking, and graph structure (Neo4j).

## Installation

### 1. Install Dependencies

```bash
pip install weaviate-client>=4.0.0 openai>=1.0.0 neo4j
```

Or install the full package:

```bash
pip install -e .
```

### 2. Start Services

**Weaviate** (vector database for embeddings):

```bash
docker run -d --name weaviate \
    -p 8081:8080 -p 50051:50051 \
    -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
    -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \
    semitechnologies/weaviate:latest
```

**Neo4j** (graph database for connections):

```bash
docker run -d --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password123 \
    neo4j:latest
```

### 3. Set Environment Variables

```bash
# Required: OpenAI API key for embeddings and reranking
export OPENAI_API_KEY="sk-..."

# Neo4j connection
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password123"

# Weaviate connection
export WEAVIATE_URL="http://localhost:8081"
```

## Usage

### Index Wiki Pages (First Time)

```python
from kapso.knowledge.search import KnowledgeSearchFactory, KGIndexInput

# Create search backend
search = KnowledgeSearchFactory.create("kg_graph_search")

# Index from wiki directory
search.index(KGIndexInput(
    wiki_dir="data/wikis",
    persist_path="data/indexes/wikis.json",
))
```

### Search (Using Already Indexed Data)

Data persists in Weaviate and Neo4j. Just create and query:

```python
from kapso.knowledge.search import KnowledgeSearchFactory, KGSearchFilters, PageType

# Create search (connects to existing indexed data)
search = KnowledgeSearchFactory.create("kg_graph_search")

# Basic search
result = search.search("How to fine-tune LLM?")

# Search with filters
result = search.search(
    query="LoRA best practices",
    filters=KGSearchFilters(
        top_k=5,
        min_score=0.5,
        page_types=[PageType.HEURISTIC, PageType.WORKFLOW],
        domains=["LLMs"],
    ),
)

# Access results
for item in result:
    print(f"{item.page_title} ({item.page_type}) - Score: {item.score:.2f}")
    
    # Graph connections (from Neo4j)
    connected = item.metadata.get("connected_pages", [])
    print(f"  Connected to {len(connected)} pages")
```

## Search Pipeline

```
Query → Embedding → Weaviate Search → LLM Reranker → Graph Enrichment → Results
```

1. **Embedding**: Generate query embedding with OpenAI
2. **Weaviate Search**: Find top-2K similar pages by vector similarity
3. **LLM Reranker**: Use gpt-4.1-mini to rerank by relevance (optional)
4. **Graph Enrichment**: Add connected pages from Neo4j (optional)

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_llm_reranker` | `True` | Enable LLM-based result reranking |
| `reranker_model` | `gpt-4.1-mini` | Model for reranking |
| `include_connected_pages` | `True` | Include graph connections |
| `embedding_model` | `text-embedding-3-large` | OpenAI embedding model |
| `weaviate_collection` | `KGWikiPages` | Weaviate collection name |

### Examples

```python
# Default: reranking + graph enrichment
search = KnowledgeSearchFactory.create("kg_graph_search")

# Fast: no reranking, no graph (semantic search only)
search = KnowledgeSearchFactory.create(
    "kg_graph_search",
    params={"use_llm_reranker": False, "include_connected_pages": False}
)

# Using preset
search = KnowledgeSearchFactory.create("kg_graph_search", preset="FAST")
```

## Directory Structure

```
src/knowledge/search/
├── base.py              # Abstract classes and data structures
├── factory.py           # Factory for creating search backends
├── kg_graph_search.py   # Weaviate + Neo4j + LLM reranker (includes wiki parser)
├── kg_llm_navigation_search.py  # LLM navigation implementation (legacy)
└── knowledge_search.yaml        # Configuration presets
```

## Presets (knowledge_search.yaml)

```yaml
kg_graph_search:
  presets:
    DEFAULT:        # Full pipeline
      params:
        use_llm_reranker: true
        include_connected_pages: true
    
    FAST:           # Semantic search only
      params:
        use_llm_reranker: false
        include_connected_pages: false
    
    RERANK_ONLY:    # Reranking without graph
      params:
        use_llm_reranker: true
        include_connected_pages: false
```

## Test

```bash
# Set environment variables (or source .env)
source .env
export NEO4J_PASSWORD="password123"

# Run test
python test_search.py
```
