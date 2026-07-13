#!/bin/bash
# One-time asset build (no GPU needed): a cheap CPU VM builds the apptainer
# containers (kapso.sif, vllm_debug.sif), uploads them to GCS, and fills a
# persistent disk with the HuggingFace cache, which is then snapshotted.
# Per-run VMs later clone that snapshot instead of re-downloading anything.
#
# Duration: ~1h for containers + minutes (core) to hours (full) for the cache.

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

BUILDER="ptb-builder"

gcloud compute disks describe "$CACHE_DISK" --zone "$ZONE" --project "$PROJECT" >/dev/null 2>&1 || \
    gcloud compute disks create "$CACHE_DISK" --project "$PROJECT" --zone "$ZONE" \
        --size "${CACHE_DISK_SIZE_GB}GB" --type pd-balanced

gcloud compute instances create "$BUILDER" \
    --project "$PROJECT" --zone "$ZONE" \
    --machine-type e2-standard-16 \
    --image-family ubuntu-2204-lts --image-project ubuntu-os-cloud \
    --boot-disk-size 200GB --boot-disk-type pd-ssd \
    --disk "name=${CACHE_DISK},device-name=hfcache,mode=rw,auto-delete=no" \
    --service-account "$SA_EMAIL" --scopes cloud-platform \
    --provisioning-model=SPOT --instance-termination-action=STOP \
    --metadata-from-file startup-script=builder_startup.sh \
    --metadata "ptb_bucket=${BUCKET},ptb_repo=${PTB_REPO_URL},kapso_repo=${KAPSO_REPO_URL},cache_scope=${CACHE_SCOPE}"

echo "Builder started. Waiting for gs://$BUCKET/assets/BUILD_DONE ..."
echo "(follow along: gcloud compute instances tail-serial-port-output $BUILDER --zone $ZONE)"

for _ in $(seq 1 720); do   # up to 12h
    if gsutil -q stat "gs://$BUCKET/assets/BUILD_DONE" 2>/dev/null; then
        echo "Build finished."
        break
    fi
    sleep 60
done
gsutil -q stat "gs://$BUCKET/assets/BUILD_DONE" || { echo "Timed out; inspect the builder VM."; exit 1; }

gcloud compute instances delete "$BUILDER" --zone "$ZONE" --project "$PROJECT" --quiet

gcloud compute snapshots describe "$CACHE_SNAPSHOT" --project "$PROJECT" >/dev/null 2>&1 && \
    gcloud compute snapshots delete "$CACHE_SNAPSHOT" --project "$PROJECT" --quiet
gcloud compute disks snapshot "$CACHE_DISK" --project "$PROJECT" --zone "$ZONE" \
    --snapshot-names "$CACHE_SNAPSHOT"

echo "Assets ready: gs://$BUCKET/assets/{kapso.sif,vllm_debug.sif}, snapshot $CACHE_SNAPSHOT"
