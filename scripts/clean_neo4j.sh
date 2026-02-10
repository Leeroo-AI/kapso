#!/bin/bash
# =============================================================================
# Clean Neo4j — Removes all WikiPage nodes and relationships
#
# Usage: ./scripts/clean_neo4j.sh
#
# Connects to Neo4j at bolt://localhost:7687 (or NEO4J_URI env var).
# Deletes ALL WikiPage nodes and their relationships.
# =============================================================================

set -e
cd "$(dirname "$0")/.."

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

# Connection defaults (same as kg_graph_search.py)
NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-password}"

# Extract host:port from URI for the HTTP API
# Neo4j HTTP API runs on port 7474 by default
NEO4J_HTTP="${NEO4J_HTTP:-http://localhost:7474}"

echo -e "${CYAN}=========================================="
echo "  Neo4j Cleanup"
echo -e "==========================================${NC}"
echo ""
echo "  URI: ${NEO4J_URI}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Check Neo4j is reachable
# ─────────────────────────────────────────────────────────────────────────────
if ! curl -s "${NEO4J_HTTP}" > /dev/null 2>&1; then
    echo -e "${RED}Error: Neo4j is not reachable at ${NEO4J_HTTP}${NC}"
    echo "  Start it with: ./scripts/start_infra.sh"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Helper: run a Cypher query via the Neo4j HTTP API
# ─────────────────────────────────────────────────────────────────────────────
run_cypher() {
    local query="$1"
    curl -s \
        -u "${NEO4J_USER}:${NEO4J_PASSWORD}" \
        -H "Content-Type: application/json" \
        -d "{\"statements\": [{\"statement\": \"${query}\"}]}" \
        "${NEO4J_HTTP}/db/neo4j/tx/commit"
}

# ─────────────────────────────────────────────────────────────────────────────
# Count existing nodes
# ─────────────────────────────────────────────────────────────────────────────
echo -n "Counting WikiPage nodes... "
COUNT_RESPONSE=$(run_cypher "MATCH (p:WikiPage) RETURN count(p) AS cnt")
NODE_COUNT=$(echo "$COUNT_RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
rows = data.get('results', [{}])[0].get('data', [])
print(rows[0]['row'][0] if rows else 0)
" 2>/dev/null || echo "?")

echo -e "${YELLOW}${NODE_COUNT} nodes found${NC}"

if [ "$NODE_COUNT" = "0" ]; then
    echo -e "\n${GREEN}Neo4j is already clean. Nothing to do.${NC}"
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# Confirm
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${RED}This will delete ALL ${NODE_COUNT} WikiPage nodes and their relationships.${NC}"
read -p "Continue? (y/N): " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# Delete all WikiPage nodes and relationships
# ─────────────────────────────────────────────────────────────────────────────
echo -n "Deleting all WikiPage nodes and relationships... "
DELETE_RESPONSE=$(run_cypher "MATCH (p:WikiPage) DETACH DELETE p")

# Check for errors in response
ERRORS=$(echo "$DELETE_RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
errors = data.get('errors', [])
print(errors[0]['message'] if errors else '')
" 2>/dev/null || echo "unknown error")

if [ -n "$ERRORS" ]; then
    echo -e "${RED}Error: ${ERRORS}${NC}"
    exit 1
fi

echo -e "${GREEN}done${NC}"

# ─────────────────────────────────────────────────────────────────────────────
# Verify
# ─────────────────────────────────────────────────────────────────────────────
echo -n "Verifying... "
VERIFY_RESPONSE=$(run_cypher "MATCH (p:WikiPage) RETURN count(p) AS cnt")
REMAINING=$(echo "$VERIFY_RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
rows = data.get('results', [{}])[0].get('data', [])
print(rows[0]['row'][0] if rows else '?')
" 2>/dev/null || echo "?")

if [ "$REMAINING" = "0" ]; then
    echo -e "${GREEN}0 nodes remaining${NC}"
else
    echo -e "${YELLOW}${REMAINING} nodes remaining (may need another pass for large graphs)${NC}"
fi

echo -e "\n${GREEN}Neo4j cleanup complete.${NC}"
