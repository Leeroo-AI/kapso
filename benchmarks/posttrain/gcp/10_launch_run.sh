#!/bin/bash
# Launch one PostTrainBench run (one benchmark x one base model) on a fresh
# single-H100 VM (a3-highgpu-1g). Default provisioning is DWS Flex-start:
# the request queues until capacity is granted, then runs UNPREEMPTED to
# completion (up to --max-run-duration) at ~53% off on-demand. --spot is
# cheaper still but can be preempted mid-run — fine for dev, bad for scoring.
#
# Usage:
#   bash 10_launch_run.sh gsm8k Qwen/Qwen3-4B-Base
#   bash 10_launch_run.sh aime2025 Qwen/Qwen3-1.7B-Base --hours 10 \
#       --agent-config claude-opus-4-6 --spot

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

EVAL="${1:?usage: 10_launch_run.sh <eval> <model> [--hours N] [--agent-config M] [--spot] [--max-run-duration D]}"
MODEL="${2:?missing model (e.g. Qwen/Qwen3-4B-Base)}"
shift 2

HOURS=10
AGENT_CONFIG="claude-opus-4-6"
SPOT=0
# 10h agent + eval retries + contamination judge + uploads, with margin.
MAX_RUN_DURATION="18h"
while [ $# -gt 0 ]; do
    case "$1" in
        --hours) HOURS="$2"; shift 2 ;;
        --agent-config) AGENT_CONFIG="$2"; shift 2 ;;
        --spot) SPOT=1; shift ;;
        --max-run-duration) MAX_RUN_DURATION="$2"; shift 2 ;;
        *) echo "unknown flag: $1" >&2; exit 1 ;;
    esac
done

SLUG=$(echo "${EVAL}-$(basename "$MODEL")" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9-' '-' )
RUN_ID="${SLUG%-}-$(date +%m%d%H%M)"
RUN_ID="${RUN_ID:0:40}"
VM="ptb-${RUN_ID}"

gcloud compute disks create "ptb-cache-${RUN_ID}" \
    --project "$PROJECT" --zone "$ZONE" \
    --source-snapshot "$CACHE_SNAPSHOT" --type pd-balanced

if [ "$SPOT" = 1 ]; then
    PROVISIONING=(--provisioning-model=SPOT)
else
    PROVISIONING=(--provisioning-model=FLEX_START --request-valid-for-duration=2h)
fi

# Golden image (02_build_image.sh) if present, else vanilla ubuntu + boot-time
# installs (run_startup.sh handles both).
if gcloud compute images describe-from-family ptb-runner --project "$PROJECT" >/dev/null 2>&1; then
    IMAGE_ARGS=(--image-family ptb-runner --image-project "$PROJECT")
else
    IMAGE_ARGS=(--image-family ubuntu-2204-lts --image-project ubuntu-os-cloud)
fi

# Notes:
#  - a3-highgpu-1g bundles its H100 (no --accelerator flag; 2x375GB local SSD
#    included — verified empirically) and requires gVNIC.
gcloud compute instances create "$VM" \
    --project "$PROJECT" --zone "$ZONE" \
    --machine-type a3-highgpu-1g \
    "${PROVISIONING[@]}" \
    --max-run-duration="$MAX_RUN_DURATION" \
    --instance-termination-action=DELETE \
    --maintenance-policy=TERMINATE \
    --reservation-affinity=none \
    --network-interface nic-type=GVNIC \
    "${IMAGE_ARGS[@]}" \
    --boot-disk-size 300GB --boot-disk-type pd-ssd \
    --disk "name=ptb-cache-${RUN_ID},device-name=hfcache,mode=rw,auto-delete=yes" \
    --service-account "$SA_EMAIL" --scopes cloud-platform \
    --metadata-from-file startup-script=run_startup.sh \
    --metadata "ptb_eval=${EVAL},ptb_model=${MODEL},ptb_hours=${HOURS},ptb_agent_config=${AGENT_CONFIG},ptb_bucket=${BUCKET},ptb_repo=${PTB_REPO_URL},kapso_repo=${KAPSO_REPO_URL},ptb_run_id=${RUN_ID}"

cat <<EOF

Launched: $VM  (run id: $RUN_ID)
Flex-start requests can sit queued until capacity frees up (check STATUS):
  gcloud compute instances describe $VM --zone $ZONE --format='value(status)'
Follow the run:
  gcloud compute instances tail-serial-port-output $VM --zone $ZONE
Results stream to:
  gs://$BUCKET/results/$RUN_ID/   (fetch with 20_fetch_results.sh $RUN_ID)
The VM deletes itself when done (or at $MAX_RUN_DURATION at the latest).
EOF
