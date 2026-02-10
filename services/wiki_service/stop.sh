#!/bin/bash
# Stop local wiki (keeps data in Docker volumes)
# Usage: ./stop.sh

set -euo pipefail
cd "$(dirname "$0")"

echo "Stopping wiki..."
docker compose down

echo ""
echo "Wiki stopped."
echo "   Data is preserved in Docker volume."
echo "   Run ./start.sh to restart."

