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
# System python3 owns torch on this image; kapso runs off PYTHONPATH (no editable install).
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/kapso/src:$HOME/kapso"
cd "$HOME/kapso"

# bootstrap.sh writes this; running setup_box.sh alone leaves it missing and the
# campaign dies mid-ideation with "Claude Code OAuth credentials not found".
# Fail here instead, before any compute is spent.
grep -q '^CLAUDE_CODE_OAUTH_TOKEN=' "$HOME/kapso/.env" 2>/dev/null || {
  echo "FATAL: $HOME/kapso/.env missing CLAUDE_CODE_OAUTH_TOKEN — run bootstrap.sh first" >&2
  exit 1; }

# Every line is stamped with UTC wall-clock. The strategy's own log lines carry
# no time of their own, so without this a finished run cannot say how long any
# phase took — only the durations a component happens to print itself.
stamp() { awk '{ "date -u +%H:%M:%S" | getline t; close("date -u +%H:%M:%S");
                 print t " " $0; fflush() }'; }

# The console stream is the SUPERSET of the run, and it is persisted HERE so the
# trace is a property of the run rather than of whatever wrapper invoked it — a
# run whose stdout nobody captured used to lose every console-only session for
# good. It is also the raw material for IOAI execution-trace verification.
#
# The drain trap is load-bearing: `exec > >(...)` alone loses whatever is still
# in flight when the script exits (measured: 624 of 2501 lines, final line gone),
# which would silently truncate the very trace this exists to keep. Closing the
# fds and waiting for the writer drains it, on the failure path too.
export PYTHONUNBUFFERED=1          # file reflects real time, not flush boundaries
RUN_LOG="${RUN_LOG:-$ROOT/run.log}"
mkdir -p "$(dirname "$RUN_LOG")"
exec > >(stamp | tee -a "$RUN_LOG"); LOG_WRITER_PID=$!
exec 2>&1
trap 'exec 1>&- 2>&-; wait $LOG_WRITER_PID 2>/dev/null || true' EXIT
echo "=== run log: $RUN_LOG ==="

echo "=== preflight: $URL  ->  $ROOT ==="
python3 -m benchmarks.kaggle.preflight --url "$URL" --root "$ROOT"

echo "=== runner: campaign on $ROOT (k/hours from run_defaults unless overridden) ==="
python3 -m benchmarks.kaggle.runner --root "$ROOT" "$@"

echo "=== done. results: $ROOT/results.json  |  trace: $RUN_LOG ==="
