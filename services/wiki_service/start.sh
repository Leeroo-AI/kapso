#!/bin/bash
# Start local wiki - builds image and brings up containers
# Usage: ./start.sh

set -euo pipefail
cd "$(dirname "$0")"

echo "=========================================="
echo "  Local Wiki Startup"
echo "=========================================="
echo ""

# Create .env from example if missing
if [ ! -f .env ]; then
    echo "📄 Creating .env from template..."
    cp .env.example .env
    echo "✓ Created .env file"
fi

# Source env for port variable
set -a
source .env 2>/dev/null || true
set +a
PORT="${WIKI_PORT:-8080}"

# Create directories for volumes
mkdir -p images state outbox

# Build the wiki image
echo ""
echo "🔨 Building wiki image..."
docker compose build

# Start containers
echo "🚀 Starting containers..."
docker compose up -d

# Wait for wiki to be healthy
echo ""
echo "⏳ Waiting for wiki to be ready..."
echo "   (This may take 1-2 minutes on first run)"
echo ""

tries=90
while ((tries--)); do
    # Check if wiki responds with HTTP 200 or 301
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}" | grep -qE "^(200|301|302)$"; then
        echo ""
        echo "=========================================="
        echo "✅ Wiki is ready!"
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
echo "❌ Wiki did not start in time."
echo ""
echo "Check logs with: docker compose logs wiki"
exit 1

