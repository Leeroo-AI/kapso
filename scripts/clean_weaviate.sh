#!/bin/bash
# =============================================================================
# Clean Weaviate — List collections and interactively remove selected ones
#
# Usage: ./scripts/clean_weaviate.sh
#
# Connects to Weaviate at http://localhost:8080 (or WEAVIATE_URL env var).
# Lists all collections, then lets you pick which to delete.
# Options: specific numbers, "all", or "none".
# =============================================================================

set -e
cd "$(dirname "$0")/.."

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Connection default (same as kg_graph_search.py)
WEAVIATE_URL="${WEAVIATE_URL:-http://localhost:8080}"

echo -e "${CYAN}=========================================="
echo "  Weaviate Cleanup"
echo -e "==========================================${NC}"
echo ""
echo "  URL: ${WEAVIATE_URL}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Check Weaviate is reachable
# ─────────────────────────────────────────────────────────────────────────────
if ! curl -s "${WEAVIATE_URL}/v1/meta" > /dev/null 2>&1; then
    echo -e "${RED}Error: Weaviate is not reachable at ${WEAVIATE_URL}${NC}"
    echo "  Start it with: ./scripts/start_infra.sh"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Fetch all collections and their object counts
# ─────────────────────────────────────────────────────────────────────────────
echo "Fetching collections..."
echo ""

# Get collection names and counts via the Weaviate REST API
COLLECTIONS_JSON=$(curl -s "${WEAVIATE_URL}/v1/schema")

# Parse collections: name + object count
COLLECTIONS_INFO=$(echo "$COLLECTIONS_JSON" | python3 -c "
import sys, json, urllib.request

data = json.load(sys.stdin)
classes = data.get('classes', [])

if not classes:
    print('EMPTY')
    sys.exit(0)

weaviate_url = '${WEAVIATE_URL}'

for cls in sorted(classes, key=lambda c: c['class']):
    name = cls['class']
    # Fetch object count via GraphQL aggregate endpoint
    try:
        url = f'{weaviate_url}/v1/graphql'
        gql = json.dumps({'query': '{Aggregate{' + name + '{meta{count}}}}'})
        req = urllib.request.Request(url, data=gql.encode(), headers={'Content-Type': 'application/json'})
        resp = urllib.request.urlopen(req)
        obj_data = json.loads(resp.read())
        count = obj_data['data']['Aggregate'][name][0]['meta']['count']
    except Exception:
        count = '?'
    print(f'{name}|{count}')
" 2>/dev/null)

if [ "$COLLECTIONS_INFO" = "EMPTY" ]; then
    echo -e "${GREEN}No collections found. Weaviate is empty.${NC}"
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# Display collections with numbers
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${BOLD}  #   Collection                          Objects${NC}"
echo "  ─── ───────────────────────────────── ───────"

# Store names in an array
declare -a COLL_NAMES=()
INDEX=1
while IFS='|' read -r name count; do
    printf "  %-3s %-37s %s\n" "${INDEX})" "${name}" "${count}"
    COLL_NAMES+=("$name")
    INDEX=$((INDEX + 1))
done <<< "$COLLECTIONS_INFO"

TOTAL=${#COLL_NAMES[@]}
echo ""
echo "  Total: ${TOTAL} collection(s)"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Ask user which to delete
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${YELLOW}Which collections to delete?${NC}"
echo "  Enter: numbers separated by spaces (e.g. \"1 3 5\")"
echo "         \"all\"  to delete everything"
echo "         \"none\" or press Enter to cancel"
echo ""
read -p "Selection: " SELECTION

# Trim whitespace
SELECTION=$(echo "$SELECTION" | xargs)

# Handle empty / none
if [ -z "$SELECTION" ] || [ "$SELECTION" = "none" ]; then
    echo -e "\n${GREEN}No collections deleted.${NC}"
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# Build list of collections to delete
# ─────────────────────────────────────────────────────────────────────────────
declare -a TO_DELETE=()

if [ "$SELECTION" = "all" ]; then
    TO_DELETE=("${COLL_NAMES[@]}")
else
    for NUM in $SELECTION; do
        # Validate it's a number
        if ! [[ "$NUM" =~ ^[0-9]+$ ]]; then
            echo -e "${RED}Invalid input: '${NUM}' — expected a number${NC}"
            exit 1
        fi
        # Validate range
        if [ "$NUM" -lt 1 ] || [ "$NUM" -gt "$TOTAL" ]; then
            echo -e "${RED}Out of range: ${NUM} (valid: 1-${TOTAL})${NC}"
            exit 1
        fi
        TO_DELETE+=("${COLL_NAMES[$((NUM - 1))]}")
    done
fi

DELETE_COUNT=${#TO_DELETE[@]}

# ─────────────────────────────────────────────────────────────────────────────
# Confirm
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${RED}Will delete ${DELETE_COUNT} collection(s):${NC}"
for name in "${TO_DELETE[@]}"; do
    echo "  - ${name}"
done
echo ""
read -p "Continue? (y/N): " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# Delete selected collections
# ─────────────────────────────────────────────────────────────────────────────
echo ""
DELETED=0
FAILED=0
for name in "${TO_DELETE[@]}"; do
    echo -n "  Deleting ${name}... "
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE "${WEAVIATE_URL}/v1/schema/${name}")
    if [ "$HTTP_CODE" = "200" ]; then
        echo -e "${GREEN}done${NC}"
        DELETED=$((DELETED + 1))
    else
        echo -e "${RED}failed (HTTP ${HTTP_CODE})${NC}"
        FAILED=$((FAILED + 1))
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}Weaviate cleanup complete.${NC}"
echo "  Deleted: ${DELETED}"
if [ "$FAILED" -gt 0 ]; then
    echo -e "  ${RED}Failed:  ${FAILED}${NC}"
fi
