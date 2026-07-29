#!/usr/bin/env bash
# =============================================================================
# run_competition.sh — (on the box) the whole loop: URL -> preflight -> runner.
#
# Assumes the box is set up (setup_box.sh) and bootstrapped (bootstrap.sh:
# secrets present, code pulled, smoke-test passed). k and hours default to
# config run_defaults (k=8, 1.75h); pass extra runner flags after the URL.
#
#   bash run_competition.sh https://www.kaggle.com/competitions/<slug>/overview \
#        [--shared-cache-dir ~/cache] [--node-expansion 8] [--hours 1.75]
# =============================================================================
set -euo pipefail
URL="${1:?usage: run_competition.sh <kaggle-competition-url> [extra runner flags]}"
shift || true
ROOT="${ROOT:-$HOME/kaggle_run_$(date +%s)}"
# shellcheck disable=SC1091
source "$HOME/kapso-venv/bin/activate"
cd "$HOME/kapso"                 # repo root on sys.path for `python -m benchmarks...`

echo "=== preflight: $URL  ->  $ROOT ==="
python -m benchmarks.kaggle.preflight --url "$URL" --root "$ROOT"

echo "=== runner: campaign on $ROOT (k/hours from run_defaults unless overridden) ==="
python -m benchmarks.kaggle.runner --root "$ROOT" "$@"

echo "=== done. results: $ROOT/results.json ==="
