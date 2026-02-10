#!/bin/bash
# Full reset - removes all wiki data and volumes
# Usage: ./reset.sh

set -euo pipefail
cd "$(dirname "$0")"

echo "⚠️  WARNING: This will DELETE all wiki data!"
echo ""
echo "   - Database contents"
echo "   - Uploaded images"
echo "   - Indexer state"
echo "   - Sync state"
echo ""
read -r -p "Are you sure? [y/N] " ans
case "${ans:-}" in
    y|Y|yes|YES) ;;
    *) echo "Aborted."; exit 1 ;;
esac

echo ""
echo "Removing containers and volumes..."
docker compose down -v --remove-orphans

echo "Clearing local directories..."
rm -rf images/* state/* outbox/* 2>/dev/null || true

echo "🗑️  Clearing sync data (state + conflicts)..."
rm -f state/sync.json 2>/dev/null || true
rm -rf ../../data/wikis/_conflicts 2>/dev/null || true

echo "🗑️  Removing import flag (so pages reimport on next start)..."
rm -f ../../data/wikis/.wikis_imported 2>/dev/null || true

echo ""
echo "✅ Wiki reset complete."
echo "   Run ./start.sh to create a fresh wiki."

