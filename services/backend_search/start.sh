#!/bin/bash

# Start Backend Search Service
# Usage: ./start.sh [--port PORT]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KAPSO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

# Default port
PORT=${PORT:-3003}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Load environment – root .env first, then local .env can override
if [ -f "$KAPSO_ROOT/.env" ]; then
    export $(grep -v '^#' "$KAPSO_ROOT/.env" | xargs)
fi
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Check for KG_INDEX_PATH
if [ -z "$KG_INDEX_PATH" ]; then
    echo "Warning: KG_INDEX_PATH not set, search will use defaults"
fi

# Add paths so 'backend_search' and 'kapso' modules are found
export PYTHONPATH="${SCRIPT_DIR}/..:${KAPSO_ROOT}/src:${PYTHONPATH}"

echo "Starting Backend Search Service on port $PORT..."
echo "PYTHONPATH includes: ${SCRIPT_DIR}/.., ${KAPSO_ROOT}/src"

# Run with uvicorn in background
nohup uvicorn backend_search.app:app \
    --host 0.0.0.0 \
    --port $PORT \
    > backend_search.log 2>&1 &

echo $! > backend_search.pid
echo "Backend Search started with PID $(cat backend_search.pid)"
echo "Logs: tail -f backend_search.log"
