#!/bin/bash
# =============================================================================
# Tinkerer Infrastructure Stop/Kill Script
# - Stops all containers
# - Optionally removes all data (--volumes)
# =============================================================================

# Navigate to project root (parent of scripts directory)
cd "$(dirname "$0")/.."

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

COMPOSE_FILE="services/infrastructure/docker-compose.yml"
REMOVE_VOLUMES=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--volumes)
            REMOVE_VOLUMES=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -v, --volumes   Remove all volumes (wipe all data)"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Determine if we need sudo for docker
DOCKER_CMD="docker"
if ! docker info &> /dev/null 2>&1; then
    if sudo docker info &> /dev/null 2>&1; then
        DOCKER_CMD="sudo docker"
    fi
fi

echo -e "${YELLOW}Stopping Tinkerer infrastructure...${NC}"

if [[ "$REMOVE_VOLUMES" == "true" ]]; then
    echo -e "${RED}WARNING: This will DELETE all data:${NC}"
    echo "  - Neo4j graph database"
    echo "  - Weaviate vector database"
    echo "  - MediaWiki pages and database"
    echo ""
    
    if [[ -t 0 ]]; then
        read -p "Are you sure? [y/N]: " confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 0
        fi
    fi
    
    $DOCKER_CMD compose -f "$COMPOSE_FILE" down -v --remove-orphans
    
    # Also clean up local index files
    rm -f data/indexes/wikis.json 2>/dev/null
    rm -f data/wikis/.wikis_imported 2>/dev/null
    
    echo -e "${GREEN}All containers and data removed.${NC}"
else
    $DOCKER_CMD compose -f "$COMPOSE_FILE" down
    echo -e "${GREEN}Containers stopped. Data preserved in volumes.${NC}"
fi

echo ""
echo "To restart: ./scripts/start_infra.sh"
