# Infrastructure Setup

Unified scripts and Docker configuration for Praxium's knowledge infrastructure.

## Components

| Service | Port | Purpose |
|---------|------|---------|
| **Weaviate** | 8080 | Vector database for semantic search |
| **Neo4j** | 7474 (HTTP), 7687 (Bolt) | Graph database for knowledge relationships |
| **MediaWiki** | 8090 | Wiki interface for knowledge content |
| **MariaDB** | 3306 | Database backend for MediaWiki |

## Quick Start

```bash
# Start all services with interactive wiki data selection
./start_infra.sh

# Start with specific wiki directory
./start_infra.sh --wiki-dir data/wikis_batch_top100

# Skip KG indexing (faster, but no search)
./start_infra.sh --skip-index

# Force reimport even if already done
./start_infra.sh --force
```

## Stopping Services

```bash
# Stop containers (preserves data)
./stop_infra.sh

# Stop and remove ALL data (clean slate)
./stop_infra.sh --volumes
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    start_infra.sh                           │
├─────────────────────────────────────────────────────────────┤
│  1. Docker Compose Up (Weaviate, Neo4j, MediaWiki, MariaDB) │
│  2. Wait for services to be ready                           │
│  3. Interactive: Select wiki data directory                 │
│  4. Import .md files to MediaWiki via API                   │
│  5. Index wiki pages to Weaviate + Neo4j for search         │
└─────────────────────────────────────────────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Multi-service container configuration |
| `start_infra.sh` | Master startup + data import + indexing |
| `stop_infra.sh` | Shutdown with optional data cleanup |

## Service URLs (After Start)

- **MediaWiki**: http://localhost:8090
- **Weaviate**: http://localhost:8080/v1/meta
- **Neo4j Browser**: http://localhost:7474 (user: `neo4j`, password: `password`)

## Wiki Data

The startup script prompts you to select from available wiki directories:
- `data/wikis/` - Full wiki content
- `data/wikis_batch_top100/` - Top 100 most important pages

## Troubleshooting

### Docker Permission Denied
```bash
sudo usermod -aG docker $USER
# Then log out and back in
```

### Services Not Starting
```bash
cd services/infrastructure
docker compose logs weaviate
docker compose logs neo4j
```

## Environment Variables

Set in `.env`:
- `OPENAI_API_KEY` - Required for embeddings during KG indexing
