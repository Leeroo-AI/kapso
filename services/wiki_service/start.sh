#!/bin/bash
# Start wiki service - builds image and brings up containers
# Usage: ./start.sh [--wiki-dir PATH] [--port PORT] [--fresh]
#
# Options:
#   --wiki-dir PATH   Path to wiki data directory (relative to project root)
#                     Default: data/wikis
#   --port PORT       Starting port number (will auto-increment if in use)
#                     Default: 8080
#   --fresh           Force reimport of wiki pages (removes import flag)
#
# Examples:
#   ./start.sh                                    # Uses data/wikis on port 8080
#   ./start.sh --wiki-dir data/wikis_llm_finetuning
#   ./start.sh --wiki-dir data/wikis_llm_finetuning --fresh
#   ./start.sh --port 8090

set -euo pipefail
cd "$(dirname "$0")"

# Function to check if a port is available
is_port_available() {
    local port=$1
    # Check if anything is listening on the port
    if command -v ss &> /dev/null; then
        ! ss -tuln | grep -q ":${port} "
    elif command -v netstat &> /dev/null; then
        ! netstat -tuln | grep -q ":${port} "
    else
        # Fallback: try to connect and see if it fails
        ! (echo >/dev/tcp/localhost/$port) 2>/dev/null
    fi
}

# Function to find an available port starting from a given port
find_available_port() {
    local start_port=$1
    local max_tries=${2:-10}
    local port=$start_port

    for ((i=0; i<max_tries; i++)); do
        if is_port_available $port; then
            echo $port
            return 0
        fi
        echo "Warning: Port $port is in use, trying next..." >&2
        port=$((port + 1))
    done

    echo "Error: Could not find available port after $max_tries attempts" >&2
    return 1
}

# Parse command line arguments
WIKI_DIR_ARG=""
PORT_ARG=""
FRESH_IMPORT=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --wiki-dir)
            WIKI_DIR_ARG="$2"
            shift 2
            ;;
        --port)
            PORT_ARG="$2"
            shift 2
            ;;
        --fresh)
            FRESH_IMPORT=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./start.sh [--wiki-dir PATH] [--port PORT] [--fresh]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "  Wiki Service"
echo "=========================================="
echo ""

# Create .env from example if missing
if [ ! -f .env ]; then
    echo "Creating .env from template..."
    cp .env.example .env 2>/dev/null || touch .env
    echo "Created .env file"
fi

# Source env for port variable
set -a
source .env 2>/dev/null || true
set +a

# Determine starting port (command line arg > env var > default)
if [ -n "$PORT_ARG" ]; then
    START_PORT="$PORT_ARG"
else
    START_PORT="${WIKI_PORT:-8080}"
fi

# Find an available port
echo "Checking port availability..."
PORT=$(find_available_port $START_PORT)
if [ $? -ne 0 ]; then
    echo "Error: Failed to find an available port"
    exit 1
fi

# Export the port for docker-compose
export WIKI_PORT="$PORT"
echo "Using wiki port: $PORT"

# Set wiki directory (command line arg > env var > default)
if [ -n "$WIKI_DIR_ARG" ]; then
    export WIKI_DIR="$WIKI_DIR_ARG"
elif [ -z "${WIKI_DIR:-}" ]; then
    export WIKI_DIR="data/wikis"
fi

echo "Wiki data directory: $WIKI_DIR"

# Remove import flag if --fresh specified (forces reimport)
if [ "$FRESH_IMPORT" = true ]; then
    IMPORT_FLAG="../../${WIKI_DIR}/.wikis_imported"
    if [ -f "$IMPORT_FLAG" ]; then
        echo "Removing import flag for fresh import..."
        rm -f "$IMPORT_FLAG"
    fi
fi

# Create directories for volumes
mkdir -p images state outbox

# Build the wiki image
echo ""
echo "Building wiki image..."
docker compose build wiki

# Start containers
echo "Starting containers..."
docker compose up -d

# Wait for wiki to be healthy
echo ""
echo "Waiting for wiki to be ready..."
echo "   (This may take 1-2 minutes on first run)"
echo ""

tries=90
while ((tries--)); do
    # Check if wiki responds with HTTP 200 or 301/302
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}" | grep -qE "^(200|301|302)$"; then
        echo ""
        echo "=========================================="
        echo "Wiki is ready!"
        echo ""
        echo "   URL:      http://localhost:${PORT}"
        echo ""
        echo "   Admin:    ${MW_ADMIN_USER:-admin}"
        echo "   Password: ${MW_ADMIN_PASS:-adminpass123}"
        echo ""
        echo "   API User: ${MW_AGENT_USER:-agent}"
        echo "   API Pass: ${MW_AGENT_PASS:-agentpass123}"
        echo "=========================================="
        echo ""
        echo "Commands:"
        echo "  ./stop.sh   - Stop wiki (keeps data)"
        echo "  ./reset.sh  - Delete all data"
        echo "  docker compose logs -f wiki - View logs"
        echo ""
        exit 0
    fi
    printf "."
    sleep 2
done

echo ""
echo "Error: Wiki did not start in time."
echo ""
echo "Check logs with: docker compose logs wiki"
exit 1
