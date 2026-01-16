#!/bin/bash
# =============================================================================
# Kapso Infrastructure Startup Script
# - Starts Weaviate, Neo4j, MediaWiki containers
# - Optionally imports wiki pages to MediaWiki (for browsing)
#
# Note: For KG indexing, use Kapso.index_kg() in Python:
#   kapso = Kapso(config_path="src/config.yaml")
#   kapso.index_kg(wiki_dir="data/wikis_llm_finetuning", save_to="data/indexes/my.index")
# =============================================================================

set -e
# Navigate to project root (parent of scripts directory)
cd "$(dirname "$0")/.."

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

WIKI_DIR=""
SKIP_INDEX=false
FORCE_IMPORT=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --wiki-dir)
            WIKI_DIR="$2"
            shift 2
            ;;
        --skip-index)
            SKIP_INDEX=true
            shift
            ;;
        --force)
            FORCE_IMPORT=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --wiki-dir PATH   Wiki data directory"
            echo "  --skip-index      Skip wiki import and indexing"
            echo "  --force           Force reimport even if pages exist"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}=========================================="
echo "  Kapso Infrastructure Startup"
echo -e "==========================================${NC}"

# Check prerequisites
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker not found. Please install Docker first.${NC}"
    exit 1
fi

# Determine if we need sudo for docker
DOCKER_CMD="docker"
if ! docker info &> /dev/null 2>&1; then
    if sudo docker info &> /dev/null 2>&1; then
        DOCKER_CMD="sudo docker"
        echo -e "${YELLOW}Note: Using sudo for Docker commands${NC}"
    else
        echo -e "${RED}Docker daemon not accessible. Check permissions.${NC}"
        exit 1
    fi
fi

# =============================================================================
# Step 1: Select wiki data directory (interactive if not specified)
# =============================================================================
if [[ -z "$WIKI_DIR" ]] && [[ "$SKIP_INDEX" == "false" ]]; then
    echo -e "\n${YELLOW}Available wiki data directories:${NC}"
    
    wiki_dirs=($(ls -d data/wikis* 2>/dev/null | grep -v "_staging" | grep -v "\."))
    
    if [[ ${#wiki_dirs[@]} -eq 0 ]]; then
        echo "  No wiki directories found in data/"
        SKIP_INDEX=true
    else
        for i in "${!wiki_dirs[@]}"; do
            count=$(find "${wiki_dirs[$i]}" -maxdepth 2 -name "*.md" 2>/dev/null | wc -l)
            echo "  [$((i+1))] ${wiki_dirs[$i]} ($count .md files)"
        done
        echo "  [0] Skip wiki import"
        echo ""
        
        if [[ -t 0 ]]; then
            read -p "Select wiki directory [1]: " choice
        else
            choice=1
        fi
        
        if [[ -z "$choice" ]]; then
            choice=1
        fi
        
        if [[ "$choice" == "0" ]]; then
            SKIP_INDEX=true
            echo "Skipping wiki import."
        elif [[ "$choice" -ge 1 ]] && [[ "$choice" -le ${#wiki_dirs[@]} ]]; then
            WIKI_DIR="${wiki_dirs[$((choice-1))]}"
            echo -e "Selected: ${GREEN}$WIKI_DIR${NC}"
        else
            echo "Invalid choice, using default"
            WIKI_DIR="${wiki_dirs[0]}"
        fi
    fi
fi

# =============================================================================
# Step 2: Start Docker containers
# =============================================================================
COMPOSE_FILE="services/infrastructure/docker-compose.yml"

echo -e "\n${YELLOW}Starting Docker containers...${NC}"
echo "  - Weaviate (vector DB) on :8080"
echo "  - Neo4j (graph DB) on :7474/:7687"  
echo "  - MediaWiki (web UI) on :8090"
echo "  - MariaDB (MediaWiki backend)"

$DOCKER_CMD compose -f "$COMPOSE_FILE" up -d

# =============================================================================
# Step 3: Wait for services
# =============================================================================
echo -e "\n${YELLOW}Waiting for services...${NC}"

echo -n "  Weaviate: "
for i in $(seq 1 30); do
    if curl -s "http://localhost:8080/v1/meta" > /dev/null 2>&1; then
        echo -e "${GREEN}ready${NC}"
        break
    fi
    if [[ $i -eq 30 ]]; then echo -e "${RED}timeout${NC}"; fi
    echo -n "."
    sleep 2
done

echo -n "  Neo4j: "
for i in $(seq 1 30); do
    if curl -s "http://localhost:7474" > /dev/null 2>&1; then
        echo -e "${GREEN}ready${NC}"
        break
    fi
    if [[ $i -eq 30 ]]; then echo -e "${RED}timeout${NC}"; fi
    echo -n "."
    sleep 2
done

echo -n "  MediaWiki: "
for i in $(seq 1 30); do
    if curl -s "http://localhost:8090/api.php" > /dev/null 2>&1; then
        echo -e "${GREEN}ready${NC}"
        break
    fi
    if [[ $i -eq 30 ]]; then echo -e "${RED}timeout${NC}"; fi
    echo -n "."
    sleep 2
done

# =============================================================================
# Step 4: Import wiki pages to MediaWiki
# =============================================================================
if [[ "$SKIP_INDEX" == "false" ]] && [[ -n "$WIKI_DIR" ]] && [[ -d "$WIKI_DIR" ]]; then
    echo -e "\n${YELLOW}Importing wiki pages to MediaWiki...${NC}"
    
    FORCE_FLAG=""
    if [[ "$FORCE_IMPORT" == "true" ]]; then
        FORCE_FLAG="--force"
    fi
    
    if [[ -f "services/wiki_service/tools/import_wiki_pages.py" ]]; then
        python services/wiki_service/tools/import_wiki_pages.py \
            --wiki-dir "$WIKI_DIR" \
            --base http://localhost:8090 \
            $FORCE_FLAG
    else
        echo -e "${RED}  Import script not found!${NC}"
    fi
fi

# =============================================================================
# Summary
# =============================================================================
echo -e "\n${GREEN}=========================================="
echo "  Infrastructure Ready!"
echo -e "==========================================${NC}"
echo ""
echo "  Services:"
echo "    MediaWiki:  http://localhost:8090 (admin/adminpass123)"
echo "    Neo4j:      http://localhost:7474 (neo4j/password)"
echo "    Weaviate:   http://localhost:8080"
echo ""

if [[ "$SKIP_INDEX" == "false" ]] && [[ -n "$WIKI_DIR" ]]; then
    page_count=$(curl -s "http://localhost:8090/api.php?action=query&meta=siteinfo&siprop=statistics&format=json" 2>/dev/null | grep -o '"pages":[0-9]*' | grep -o '[0-9]*' || echo "?")
    echo "  Wiki Data:    $WIKI_DIR"
    echo "  Pages:        $page_count (in MediaWiki)"
    echo ""
fi

echo "  Index KG:     kapso.index_kg(wiki_dir='...', save_to='...')"
echo ""
echo "  Stop:         ./scripts/stop_infra.sh"
echo "  Stop + wipe:  ./scripts/stop_infra.sh --volumes"
