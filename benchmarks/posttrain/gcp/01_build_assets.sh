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

# Ship the exact local kapso tree (this checkout, committed or not) so the
# container gets our branch without requiring a push.
KAPSO_ROOT="$(cd ../../.. && pwd)"
tar -czf - -C "$KAPSO_ROOT" \
    --exclude=.git --exclude=.claude --exclude=archive --exclude=tests \
    --exclude=moltbook_bot --exclude=tmp \
    --exclude='.env' --exclude='*.env' --exclude='*.pem' --exclude='*token*' . \
    | gsutil cp - "gs://$BUCKET/assets/kapso-src.tgz"

# Stale markers from a previous attempt would end the poll loop instantly.
gsutil -q rm "gs://$BUCKET/assets/BUILD_DONE" "gs://$BUCKET/assets/BUILD_FAILED" 2>/dev/null || true

gcloud compute disks describe "$CACHE_DISK" --zone "$ZONE" --project "$PROJECT" >/dev/null 2>&1 || \
    gcloud compute disks create "$CACHE_DISK" --project "$PROJECT" --zone "$ZONE" \
        --size "${CACHE_DISK_SIZE_GB}GB" --type pd-balanced

# Leftover builder from a previous (failed/preempted) attempt blocks creation.
gcloud compute instances delete "$BUILDER" --zone "$ZONE" --project "$PROJECT" --quiet 2>/dev/null || true

# On-demand only: a ~$1 build is not worth spot preemption roulette (measured
# us-central1-a spot: preempted 4m42s into a 1h build).
CREATED=0
for MACHINE_ARGS in \
    "--machine-type=e2-standard-16" \
    "--machine-type=e2-standard-8"; do
    # shellcheck disable=SC2086
    if gcloud compute instances create "$BUILDER" \
        --project "$PROJECT" --zone "$ZONE" \
        $MACHINE_ARGS \
        --image-family ubuntu-2204-lts --image-project ubuntu-os-cloud \
        --boot-disk-size 200GB --boot-disk-type pd-ssd \
        --disk "name=${CACHE_DISK},device-name=hfcache,mode=rw,auto-delete=no" \
        --service-account "$SA_EMAIL" --scopes cloud-platform \
        --metadata-from-file startup-script=builder_startup.sh \
        --metadata "ptb_bucket=${BUCKET},ptb_repo=${PTB_REPO_URL},kapso_repo=${KAPSO_REPO_URL},cache_scope=${CACHE_SCOPE},kapso_src_gcs=gs://${BUCKET}/assets/kapso-src.tgz"; then
        CREATED=1; break
    fi
    echo "builder create failed with: $MACHINE_ARGS — trying next config"
done
[ "$CREATED" = 1 ] || { echo "builder creation failed in every config"; exit 1; }

echo "Builder started. Waiting for gs://$BUCKET/assets/BUILD_DONE ..."
echo "(follow along: gcloud compute instances tail-serial-port-output $BUILDER --zone $ZONE)"

for _ in $(seq 1 720); do   # up to 12h
    if gsutil -q stat "gs://$BUCKET/assets/BUILD_DONE" 2>/dev/null; then
        echo "Build finished."
        break
    fi
    if gsutil -q stat "gs://$BUCKET/assets/BUILD_FAILED" 2>/dev/null; then
        echo "BUILD FAILED — log tail:"
        gsutil cat "gs://$BUCKET/assets/build.log" 2>/dev/null | tail -40
        exit 1
    fi
    STATUS=$(gcloud compute instances describe "$BUILDER" --zone "$ZONE" --project "$PROJECT" \
        --format='value(status)' 2>/dev/null || echo GONE)
    if [ "$STATUS" = "TERMINATED" ] || [ "$STATUS" = "GONE" ]; then
        echo "Builder died (status=$STATUS) without writing a marker — aborting."
        exit 1
    fi
    sleep 60
done
gsutil -q stat "gs://$BUCKET/assets/BUILD_DONE" || { echo "Timed out; inspect the builder VM."; exit 1; }

gcloud compute instances delete "$BUILDER" --zone "$ZONE" --project "$PROJECT" --quiet

gcloud compute snapshots describe "$CACHE_SNAPSHOT" --project "$PROJECT" >/dev/null 2>&1 && \
    gcloud compute snapshots delete "$CACHE_SNAPSHOT" --project "$PROJECT" --quiet
gcloud compute disks snapshot "$CACHE_DISK" --project "$PROJECT" --zone "$ZONE" \
    --snapshot-names "$CACHE_SNAPSHOT"

# The snapshot is the canonical asset; the source disk costs ~$50/mo idle.
gcloud compute disks delete "$CACHE_DISK" --zone "$ZONE" --project "$PROJECT" --quiet

echo "Assets ready: gs://$BUCKET/assets/{kapso.sif,vllm_debug.sif}, snapshot $CACHE_SNAPSHOT"
