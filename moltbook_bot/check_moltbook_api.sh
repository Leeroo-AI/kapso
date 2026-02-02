#!/bin/bash
# Moltbook API Health Check Script
# Tests the 3 broken endpoints (comments, upvote, subscribe)
# Runs every 5 minutes until they are fixed
#
# Usage:
#   ./check_moltbook_api.sh          # Run once
#   ./check_moltbook_api.sh --watch  # Run every 5 minutes until fixed
#   ./check_moltbook_api.sh --loop   # Run forever every 5 minutes

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
WATCH_MODE=false
LOOP_FOREVER=false
CHECK_INTERVAL=300  # 5 minutes in seconds

while [[ $# -gt 0 ]]; do
    case $1 in
        --watch|-w)
            WATCH_MODE=true
            shift
            ;;
        --loop|-l)
            LOOP_FOREVER=true
            shift
            ;;
        --interval|-i)
            CHECK_INTERVAL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--watch] [--loop] [--interval SECONDS]"
            exit 1
            ;;
    esac
done

# Load API key
if [ -f ~/.openclaw/.env ]; then
    source ~/.openclaw/.env
fi

if [ -z "${MOLTBOOK_API_KEY:-}" ]; then
    # Try credentials file
    if [ -f ~/.config/moltbook/credentials.json ]; then
        MOLTBOOK_API_KEY=$(python3 -c "import json; print(json.load(open('$HOME/.config/moltbook/credentials.json'))['api_key'])")
    fi
fi

if [ -z "${MOLTBOOK_API_KEY:-}" ]; then
    echo -e "${RED}ERROR: MOLTBOOK_API_KEY not found${NC}"
    echo "Set it in ~/.openclaw/.env or ~/.config/moltbook/credentials.json"
    exit 1
fi

BASE_URL="https://www.moltbook.com/api/v1"

# Main health check function
do_health_check() {

echo "========================================"
echo "Moltbook API - Broken Endpoints Check"
echo "$(date '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================"
echo ""

# Track results
PASSED=0
FAILED=0

# Function to test an endpoint
test_endpoint() {
    local name="$1"
    local endpoint="$2"
    local data="$3"
    
    response=$(curl -s -X POST "$BASE_URL$endpoint" \
        -H "Authorization: Bearer $MOLTBOOK_API_KEY" \
        -H "Content-Type: application/json" \
        -d "$data" 2>/dev/null)
    
    success=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('success', False))" 2>/dev/null || echo "false")
    error=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error', ''))" 2>/dev/null || echo "parse_error")
    
    if [ "$success" = "True" ] || [ "$success" = "true" ]; then
        echo -e "${GREEN}✓ FIXED${NC}: $name"
        PASSED=$((PASSED + 1))
        return 0
    else
        # Check if it's a rate limit (which means auth worked!)
        if [[ "$error" == *"20 seconds"* ]] || [[ "$error" == *"50 comments"* ]] || [[ "$error" == *"already subscribed"* ]]; then
            echo -e "${GREEN}✓ FIXED${NC}: $name (rate limited = auth works!)"
            PASSED=$((PASSED + 1))
            return 0
        else
            echo -e "${RED}✗ BROKEN${NC}: $name - $error"
            FAILED=$((FAILED + 1))
            return 1
        fi
    fi
}

# Get a post ID to test with
POST_ID=$(curl -s "$BASE_URL/posts?limit=1" \
    -H "Authorization: Bearer $MOLTBOOK_API_KEY" | \
    python3 -c "import sys,json; d=json.load(sys.stdin); print(d['posts'][0]['id'])" 2>/dev/null || echo "")

if [ -z "$POST_ID" ]; then
    echo -e "${RED}ERROR: Could not get post ID for testing${NC}"
    return 1 2>/dev/null || exit 1
fi

# Test the 3 broken endpoints
test_endpoint "POST /posts/{id}/comments" "/posts/$POST_ID/comments" '{"content":"API health check"}' || true
test_endpoint "POST /posts/{id}/upvote" "/posts/$POST_ID/upvote" '{}' || true
test_endpoint "POST /submolts/general/subscribe" "/submolts/general/subscribe" '{}' || true

echo ""
echo "========================================"
echo -e "Fixed: ${GREEN}$PASSED${NC}/3  |  Broken: ${RED}$FAILED${NC}/3"
echo "========================================"

# Determine overall status
if [ $FAILED -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 All endpoints fixed!${NC}"
    return 0 2>/dev/null || exit 0
else
    echo ""
    echo -e "${YELLOW}Still broken. Contact @mattprd on X to report.${NC}"
    return 1 2>/dev/null || exit 1
fi
}

# Main execution
run_check() {
    do_health_check
    return $?
}

# Single run mode
if [ "$WATCH_MODE" = false ] && [ "$LOOP_FOREVER" = false ]; then
    run_check
    exit $?
fi

# Watch mode - run until fixed
if [ "$WATCH_MODE" = true ]; then
    echo -e "${BLUE}Watch mode: Checking every $((CHECK_INTERVAL/60)) min until fixed...${NC}"
    echo -e "${BLUE}Press Ctrl+C to stop${NC}"
    echo ""
    
    CHECK_COUNT=0
    while true; do
        CHECK_COUNT=$((CHECK_COUNT + 1))
        echo -e "${BLUE}=== Check #$CHECK_COUNT ===${NC}"
        
        if run_check; then
            echo ""
            echo -e "${GREEN}🎉 All issues fixed! Stopping.${NC}"
            exit 0
        fi
        
        echo ""
        echo -e "${BLUE}Next check in $((CHECK_INTERVAL/60)) min...${NC}"
        echo ""
        sleep $CHECK_INTERVAL
    done
fi

# Loop forever mode
if [ "$LOOP_FOREVER" = true ]; then
    echo -e "${BLUE}Loop mode: Checking every $((CHECK_INTERVAL/60)) min forever...${NC}"
    echo -e "${BLUE}Press Ctrl+C to stop${NC}"
    echo ""
    
    CHECK_COUNT=0
    while true; do
        CHECK_COUNT=$((CHECK_COUNT + 1))
        echo -e "${BLUE}=== Check #$CHECK_COUNT ===${NC}"
        
        run_check || true
        
        echo ""
        echo -e "${BLUE}Next check in $((CHECK_INTERVAL/60)) min...${NC}"
        echo ""
        sleep $CHECK_INTERVAL
    done
fi
