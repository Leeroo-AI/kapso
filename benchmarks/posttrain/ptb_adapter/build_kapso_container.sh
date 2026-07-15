#!/bin/bash
# Build the kapso PostTrainBench container.
#
# Run from the root of a PostTrainBench checkout that already contains the
# adapter files (agents/kapso/solve.sh, containers/kapso.def — see
# ptb_adapter/README.md).
#
# Usage: bash build_kapso_container.sh /path/to/kapso/repo

set -euo pipefail

KAPSO_SRC="${1:?usage: build_kapso_container.sh <path-to-kapso-repo>}"

if [ ! -f containers/kapso.def ]; then
    echo "Run this from a PostTrainBench repo root containing containers/kapso.def" >&2
    exit 1
fi

# Stage kapso source into the apptainer build context (%files can only copy
# from the host build context).
rsync -a --delete \
    --exclude .git --exclude .claude --exclude archive --exclude tests \
    --exclude moltbook_bot --exclude 'tmp/' \
    --exclude '.env' --exclude '*.env' --exclude '*.pem' --exclude '*token*' \
    "${KAPSO_SRC%/}/" containers/kapso-src/

bash containers/build_container.sh kapso
echo "Built: ${POST_TRAIN_BENCH_CONTAINERS_DIR:-containers}/kapso.sif"
