# Wiki Service

A MediaWiki stack for serving and displaying wiki files from `data/wikis/`.

A self-contained MediaWiki stack with Semantic MediaWiki, PageForms, Cargo, and more.
Run a full wiki locally with a single command. **Automatically imports wiki pages from 
`data/wikis/` on startup and supports two-way sync.**

## Quick Start

```bash
# 1. Create environment file (optional - defaults work fine)
cp .env.example .env

# 2. Start the wiki (auto-imports pages from data/wikis/)
./start.sh

# 3. Access at http://localhost:8080
#    Login: admin / adminpass123
```

## Two-Way Sync with data/wikis

The wiki service automatically syncs with `data/wikis/`:

### Import (Automatic on Startup)

When the wiki starts, it automatically imports all `.mediawiki` files from `data/wikis/`:
- Files are organized by repository: `data/wikis/{repo_name}/{Page_Name}.mediawiki`
- Pages are created with titles: `{repo_name}/{Page_Name}`
- Import only runs once (flag file tracks completion)

To force reimport:
```bash
# Set FORCE_REIMPORT=true in .env or run:
docker compose exec wiki bash -c "FORCE_REIMPORT=true /import_wikis.sh"
```

### Export (On Demand)

To sync edits made in the wiki back to `data/wikis/`:

```bash
# Export all pages
python3 tools/export_wikis.py \
  --base http://localhost:8080 \
  --user agent \
  --password agentpass123 \
  --output ../../data/wikis

# Export specific repository only
python3 tools/export_wikis.py \
  --base http://localhost:8080 \
  --user agent \
  --password agentpass123 \
  --output ../../data/wikis \
  --prefix huggingface_text-generation-inference

# Dry run (see what would be exported)
python3 tools/export_wikis.py \
  --base http://localhost:8080 \
  --output ../../data/wikis \
  --dry-run
```

## Commands

| Command | Description |
|---------|-------------|
| `./start.sh` | Build and start wiki |
| `./stop.sh` | Stop wiki (keeps data) |
| `./reset.sh` | Stop and delete all data |
| `docker compose logs -f wiki` | View wiki logs |
| `docker compose logs -f db` | View database logs |

## Configuration

Edit `.env` to customize:

```ini
WIKI_PORT=8080           # Access port
MW_SITENAME=Local Wiki   # Wiki name
MW_ADMIN_USER=admin      # Admin username
MW_ADMIN_PASS=adminpass123  # Admin password
```

## Test API Access

Use the included Python client:

```bash
python3 tools/mw_client.py \
  --base http://localhost:8080 \
  --user agent \
  --password agentpass123 \
  --prefix Sandbox:Test
```

Expected output:
```
✔ login ok (API: http://localhost:8080/api.php)
✔ create Sandbox:Test_...
✔ append to Sandbox:Test_...
✔ move Sandbox:Test_... → Sandbox:Test_..._Moved
✔ delete Sandbox:Test_..._Moved
🎉 All API operations succeeded.
```

## Run Indexer (optional)

The indexer pulls recent changes for incremental indexing:

```bash
WIKI_URL=http://localhost:8080 \
MW_USER=agent \
MW_PASS=agentpass123 \
RC_STATE=./state/index_state.json \
OUTBOX_FILE=./outbox/latest.ndjson \
python3 tools/recentchanges_pull.py
```

## Included Extensions

| Extension | Purpose |
|-----------|---------|
| Semantic MediaWiki | Structured data and queries |
| PageForms | Form-based page editing |
| Cargo | Database-like tables |
| VisualEditor | WYSIWYG editing |
| SyntaxHighlight | Code highlighting |
| Math | LaTeX math rendering |
| Mermaid | Diagrams (flowcharts, sequence, etc.) |
| DynamicPageList3 | Dynamic page listings |
| Network | Interactive page relationship graphs |

## Knowledge Graph Visualization

The **Network extension** provides interactive graph visualization of page relationships.
Click any node to navigate to that page.

### Usage

Add this to any wiki page to show its link network:

```wikitext
{{#network:{{FULLPAGENAME}}}}
```

With options:

```wikitext
{{#network:Resource:allenai_allennlp/AllenNLP
|depth=2
|class=knowledge-graph
}}
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `depth` | How many link levels to include | 1 |
| `class` | CSS class for styling | - |

The graph shows:
- **Nodes** = Wiki pages (click to navigate)
- **Edges** = Links between pages
- **Drag** nodes to rearrange
- **Zoom** with scroll wheel

## Directory Structure

```
local_wiki/
├── docker-compose.yml    # Container definitions
├── .env                  # Your configuration (gitignored)
├── .env.example          # Configuration template
├── start.sh              # Start wiki
├── stop.sh               # Stop wiki
├── reset.sh              # Delete all data
├── mediawiki/            # Docker image files
│   ├── Dockerfile
│   ├── entrypoint.sh
│   └── ...
├── tools/                # Utility scripts
│   ├── mw_client.py      # API client
│   └── recentchanges_pull.py  # Indexer
├── images/               # Uploaded files
├── state/                # Indexer state
└── outbox/               # Indexer output
```

## Troubleshooting

### Wiki won't start
```bash
docker compose logs wiki
docker compose logs db
```

### Database issues
```bash
# Full reset (deletes all data)
./reset.sh
./start.sh
```

### API errors
Make sure you're using the correct credentials from `.env`.

### Change port
Edit `.env` and set `WIKI_PORT=9000` (or any free port), then restart:
```bash
./stop.sh
./start.sh
```

