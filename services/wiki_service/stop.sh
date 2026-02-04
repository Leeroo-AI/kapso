#!/bin/bash
# Stop local wiki (keeps data in Docker volumes)
# Usage: ./stop.sh

set -euo pipefail
cd "$(dirname "$0")"

echo "🛑 Stopping wiki..."
docker compose down

echo "🛑 Stopping Leeroopedia API..."
docker compose -f ../leeroopedia_service/docker-compose.yml down 2>/dev/null || true

echo ""
echo "✅ Wiki and Leeroopedia API stopped."
echo "   Data is preserved in Docker volume."
echo "   Run ./start.sh to restart."

