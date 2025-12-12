# Knowledge Search

Semantic search over wiki pages using embeddings (Weaviate) and graph structure (Neo4j).

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
# Required: OpenAI API key for embeddings
export OPENAI_API_KEY="sk-..."

# Neo4j connection
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password123"

# Weaviate connection
export WEAVIATE_URL="http://localhost:8081"
```

## Usage

### Index Wiki Pages

```python
from src.knowledge.search import KnowledgeSearchFactory, KGIndexInput

# Create search backend
search = KnowledgeSearchFactory.create("kg_graph_search", enabled=True)

# Index from wiki directory
search.index(KGIndexInput(
    wiki_dir="data/wikis",
    persist_path="data/indexes/wikis.json",
))
```

### Search (Using Already Indexed Data)

If you've already indexed, data persists in Weaviate and Neo4j. Just create the search instance and query directly:

```python
from src.knowledge.search import KnowledgeSearchFactory, KGSearchFilters, PageType

# Connect to existing indexed data (no index() call needed)
search = KnowledgeSearchFactory.create("kg_graph_search", enabled=True)

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

### Available Backends

| Backend | Description |
|---------|-------------|
| `kg_graph_search` | Weaviate embeddings + Neo4j graph (recommended) |
| `kg_llm_navigation` | Neo4j + LLM-guided navigation |

## Directory Structure

```
src/knowledge/search/
├── base.py              # Abstract classes and data structures
├── factory.py           # Factory for creating search backends
├── kg_graph_search.py   # Weaviate + Neo4j implementation (includes wiki parser)
├── kg_llm_navigation_search.py  # LLM navigation implementation
└── knowledge_search.yaml        # Configuration presets
```

## Configuration

Edit `knowledge_search.yaml` for presets:

```yaml
kg_graph_search:
  presets:
    DEFAULT:
      params:
        embedding_model: "text-embedding-3-large"
        include_connected_pages: true
    FAST:
      params:
        include_connected_pages: false
```

## Test

```bash
# Activate environment
conda activate praxium_conda

# Set environment variables (or source .env)
export OPENAI_API_KEY="sk-..."
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password123"

# Run test
python -m src.knowledge.search.kg_graph_search
```

