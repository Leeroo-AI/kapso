#!/bin/bash
# Officially score an existing run's final_model (full test set), without
# re-running the agent. Pulls final_model from the run's GCS results, runs
# the harness evaluate.py in the vllm_debug container on a flex-start H100,
# uploads metrics to gs://$BUCKET/results/<run_id>/rescore/, self-deletes.
#
# Usage: bash 40_eval_only.sh <run_id> [eval_task]   (eval_task default gsm8k)

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

RUN_ID="${1:?usage: 40_eval_only.sh <run_id> [eval_task]}"
EVAL="${2:-gsm8k}"

VM="ptb-rescore-$(date +%H%M%S)"
DISK="ptb-cache-$VM"

STARTUP=$(mktemp)
cat > "$STARTUP" <<'EOS'
#!/bin/bash
set -x
exec > /var/log/ptb-rescore.log 2>&1

meta() { curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }

BUCKET=$(meta ptb_bucket)
PTB_REPO=$(meta ptb_repo)
RUN_ID=$(meta ptb_run_id)
EVAL=$(meta ptb_eval)
OUT="gs://$BUCKET/results/$RUN_ID/rescore"

finish() {
    code=$?
    gsutil cp /var/log/ptb-rescore.log "$OUT/rescore.log" || true
    echo "exit_code=$code" | gsutil cp - "$OUT/RESCORE_DONE" || true
    NAME=$(curl -sf -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/name)
    VMZONE=$(curl -sf -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
    gcloud compute instances delete "$NAME" --zone "$VMZONE" --quiet || poweroff
}
trap finish EXIT

if [ ! -f /etc/ptb-image-ready ]; then
    export DEBIAN_FRONTEND=noninteractive
    dpkg --configure -a || true
    apt-get update
    apt-get install -y software-properties-common git python3 python-is-python3
    add-apt-repository -y ppa:apptainer/ppa
    apt-get update
    apt-get install -y apptainer fuse-overlayfs
    apt-get install -y nvidia-driver-570-server || apt-get install -y nvidia-driver-550-server
fi
for _ in $(seq 1 40); do nvidia-smi && break; sleep 15; done
nvidia-smi || exit 1

mkdir -p /mnt/hfcache
mount /dev/disk/by-id/google-hfcache /mnt/hfcache
export HF_HOME=/mnt/hfcache/huggingface

git clone --depth 1 "$PTB_REPO" /opt/ptb
cd /opt/ptb

MODEL_DIR=/opt/ptb/rescore_final_model
mkdir -p "$MODEL_DIR"
SRC=$(gsutil ls -d "gs://$BUCKET/results/$RUN_ID/results/**/final_model/" | head -1)
[ -n "$SRC" ] || { echo "no final_model found for $RUN_ID"; exit 1; }
gsutil -m rsync -r "$SRC" "$MODEL_DIR"

SIF=/mnt/hfcache/containers/vllm_debug.sif
[ -f "$SIF" ] || { gsutil cp "gs://$BUCKET/assets/vllm_debug.sif" /opt/vllm_debug.sif; SIF=/opt/vllm_debug.sif; }

run_eval() {
    local extra="$1" n="$2"
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
    sleep 5
    # shellcheck disable=SC2086
    timeout --signal=TERM --kill-after=60s 4h apptainer exec \
        --nv \
        --env HF_HOME="$HF_HOME" \
        --env VLLM_API_KEY="inspectai" \
        --env PYTHONNOUSERSITE="1" \
        --writable-tmpfs \
        --bind /opt/ptb:/opt/ptb \
        --bind /mnt/hfcache:/mnt/hfcache \
        --pwd "/opt/ptb/src/eval/tasks/$EVAL" \
        "$SIF" python evaluate.py \
            --model-path "$MODEL_DIR" \
            --templates-dir /opt/ptb/src/eval/templates \
            --limit -1 \
            --max-connections 8 \
            $extra \
            --json-output-file /opt/ptb/metrics.json > "/opt/ptb/rescore_eval_$n.txt" 2>&1
}

for attempt in 1 2; do
    [ -f /opt/ptb/metrics.json ] && break
    run_eval "" "$attempt" || true
done
[ -f /opt/ptb/metrics.json ] || run_eval "--max-tokens 3000" 3 || true

gsutil cp /opt/ptb/metrics.json "$OUT/metrics.json" || true
gsutil cp /opt/ptb/rescore_eval_*.txt "$OUT/" || true
EOS

if gcloud compute images describe-from-family ptb-runner --project "$PROJECT" >/dev/null 2>&1; then
    IMAGE_ARGS=(--image-family ptb-runner --image-project "$PROJECT")
else
    IMAGE_ARGS=(--image-family ubuntu-2204-lts --image-project ubuntu-os-cloud)
fi

gcloud compute disks create "$DISK" --project "$PROJECT" --zone "$ZONE" \
    --source-snapshot "$CACHE_SNAPSHOT" --type pd-balanced

gcloud compute instances create "$VM" \
    --project "$PROJECT" --zone "$ZONE" \
    --machine-type a3-highgpu-1g \
    --provisioning-model=FLEX_START --request-valid-for-duration=2h \
    --max-run-duration=6h \
    --instance-termination-action=DELETE \
    --maintenance-policy=TERMINATE \
    --reservation-affinity=none \
    --network-interface nic-type=GVNIC \
    "${IMAGE_ARGS[@]}" \
    --boot-disk-size 100GB --boot-disk-type pd-ssd \
    --disk "name=${DISK},device-name=hfcache,mode=rw,auto-delete=yes" \
    --service-account "$SA_EMAIL" --scopes cloud-platform \
    --metadata-from-file startup-script="$STARTUP" \
    --metadata "ptb_bucket=${BUCKET},ptb_repo=${PTB_REPO_URL},ptb_run_id=${RUN_ID},ptb_eval=${EVAL}" \
    --async
rm -f "$STARTUP"

echo "Rescore launched: $VM -> gs://$BUCKET/results/$RUN_ID/rescore/"
