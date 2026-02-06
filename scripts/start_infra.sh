#!/bin/bash
# =============================================================================
# Kapso Infrastructure Startup Script
# - Starts Weaviate and Neo4j containers
# =============================================================================

set -e
# Navigate to project root (parent of scripts directory)
cd "$(dirname "$0")/.."

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
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
# Step 1: Start Docker containers
# =============================================================================
COMPOSE_FILE="services/infrastructure/docker-compose.yml"

echo -e "\n${YELLOW}Starting Docker containers...${NC}"
echo "  - Weaviate (vector DB) on :8080"
echo "  - Neo4j (graph DB) on :7474/:7687"

$DOCKER_CMD compose -f "$COMPOSE_FILE" up -d

# =============================================================================
# Step 2: Wait for services
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

# =============================================================================
# Summary
# =============================================================================
echo -e "\n${GREEN}=========================================="
echo "  Infrastructure Ready!"
echo -e "==========================================${NC}"
echo ""
echo "  Services:"
echo "    Neo4j:      http://localhost:7474 (neo4j/password)"
echo "    Weaviate:   http://localhost:8080"
echo ""
echo "  Stop:         ./scripts/stop_infra.sh"
echo "  Stop + wipe:  ./scripts/stop_infra.sh --volumes"
