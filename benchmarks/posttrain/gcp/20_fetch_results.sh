#!/bin/bash
# Fetch and summarize one run's results from GCS.
# Usage: bash 20_fetch_results.sh <run_id>          (list runs with no args)

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

if [ $# -eq 0 ]; then
    echo "Runs in gs://$BUCKET/results/:"
    gsutil ls "gs://$BUCKET/results/" 2>/dev/null || echo "(none)"
    exit 0
fi

RUN_ID="$1"
OUT="results_fetched/$RUN_ID"
mkdir -p "$OUT"
gsutil -m rsync -r "gs://$BUCKET/results/$RUN_ID" "$OUT"

echo
echo "=== $RUN_ID ==="
[ -f "$OUT/RUN_DONE" ] && cat "$OUT/RUN_DONE"
find "$OUT" -name time_taken.txt -exec sh -c 'echo "agent time: $(cat "$1")"' _ {} \;
find "$OUT" -name metrics.json -exec sh -c 'echo "metrics ($1):"; cat "$1"; echo' _ {} \;
for judgement in contamination_judgement.txt disallowed_model_judgement.txt; do
    find "$OUT" -name "$judgement" -exec sh -c 'echo "--- $(basename "$1") ---"; cat "$1"; echo' _ {} \;
done
echo "full artifacts in: $OUT"
