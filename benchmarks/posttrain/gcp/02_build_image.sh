#!/bin/bash
# One-time (and after dependency changes): bake a "golden" runner image with
# the NVIDIA driver, apptainer, and host tools preinstalled, so run VMs boot
# straight to work (~1-2 min) instead of apt-installing every time (~6-10 min,
# and fragile against PPA/apt outages mid-campaign).
#
# The driver dkms-builds fine on a cheap CPU VM; the module loads on first
# boot with a GPU attached. Nothing secret or run-specific goes in the image.
# 10_launch_run.sh picks the image up automatically via the image family.

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

BUILDER="ptb-image-builder"
IMAGE_FAMILY="ptb-runner"
IMAGE_NAME="ptb-runner-$(date +%Y%m%d-%H%M)"
SETUP_MARKER="gs://$BUCKET/assets/IMAGE_SETUP_DONE"

gcloud compute instances delete "$BUILDER" --zone "$ZONE" --project "$PROJECT" --quiet 2>/dev/null || true
gsutil -q rm "$SETUP_MARKER" 2>/dev/null || true

SETUP=$(mktemp)
cat > "$SETUP" <<EOF
#!/bin/bash
set -x
exec > /var/log/ptb-image-setup.log 2>&1
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y software-properties-common git rsync jq python3 python-is-python3 uuid-runtime tree mdadm
add-apt-repository -y ppa:apptainer/ppa
apt-get update
apt-get install -y apptainer fuse-overlayfs
apt-get install -y nvidia-driver-570-server || apt-get install -y nvidia-driver-550-server
if dpkg -l apptainer fuse-overlayfs >/dev/null 2>&1 && ls /usr/bin/nvidia-smi >/dev/null 2>&1; then
    touch /etc/ptb-image-ready   # run_startup.sh skips installs when this exists
    echo done | gsutil cp - "$SETUP_MARKER"
fi
poweroff
EOF

# On-demand only: a preempted image build produces a silently corrupt image
# (learned the hard way: TERMINATED cannot distinguish poweroff from preemption).
gcloud compute instances create "$BUILDER" \
    --project "$PROJECT" --zone "$ZONE" \
    --machine-type e2-standard-8 \
    --image-family ubuntu-2204-lts --image-project ubuntu-os-cloud \
    --boot-disk-size 50GB --boot-disk-type pd-balanced \
    --service-account "$SA_EMAIL" --scopes cloud-platform \
    --metadata-from-file startup-script="$SETUP"
rm -f "$SETUP"

echo "Waiting for image builder to finish (marker + poweroff)..."
for _ in $(seq 1 60); do
    STATUS=$(gcloud compute instances describe "$BUILDER" --zone "$ZONE" --project "$PROJECT" \
        --format='value(status)')
    [ "$STATUS" = "TERMINATED" ] && break
    sleep 20
done
[ "$STATUS" = "TERMINATED" ] || { echo "builder did not finish; inspect $BUILDER"; exit 1; }
gsutil -q stat "$SETUP_MARKER" || {
    echo "builder terminated WITHOUT the setup marker — not imaging a broken disk"; exit 1; }

gcloud compute images create "$IMAGE_NAME" \
    --project "$PROJECT" \
    --source-disk "$BUILDER" --source-disk-zone "$ZONE" \
    --family "$IMAGE_FAMILY"

gcloud compute instances delete "$BUILDER" --zone "$ZONE" --project "$PROJECT" --quiet

echo "Image $IMAGE_NAME created in family '$IMAGE_FAMILY'."
