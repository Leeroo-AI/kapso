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

SETUP=$(mktemp)
cat > "$SETUP" <<'EOF'
#!/bin/bash
set -x
exec > /var/log/ptb-image-setup.log 2>&1
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y software-properties-common git rsync jq python3 uuid-runtime tree mdadm
add-apt-repository -y ppa:apptainer/ppa
apt-get update
apt-get install -y apptainer fuse-overlayfs
apt-get install -y nvidia-driver-570-server || apt-get install -y nvidia-driver-550-server
touch /etc/ptb-image-ready   # run_startup.sh skips installs when this exists
poweroff
EOF

CREATED=0
for MACHINE_ARGS in \
    "--machine-type=e2-standard-8 --provisioning-model=SPOT --instance-termination-action=STOP" \
    "--machine-type=e2-standard-8"; do
    # shellcheck disable=SC2086
    if gcloud compute instances create "$BUILDER" \
        --project "$PROJECT" --zone "$ZONE" \
        $MACHINE_ARGS \
        --image-family ubuntu-2204-lts --image-project ubuntu-os-cloud \
        --boot-disk-size 50GB --boot-disk-type pd-balanced \
        --metadata-from-file startup-script="$SETUP"; then
        CREATED=1; break
    fi
    echo "image-builder create failed with: $MACHINE_ARGS — trying next config"
done
rm -f "$SETUP"
[ "$CREATED" = 1 ] || { echo "image-builder creation failed in every config"; exit 1; }

echo "Waiting for image builder to power off..."
for _ in $(seq 1 60); do
    STATUS=$(gcloud compute instances describe "$BUILDER" --zone "$ZONE" --project "$PROJECT" \
        --format='value(status)')
    [ "$STATUS" = "TERMINATED" ] && break
    sleep 20
done
[ "$STATUS" = "TERMINATED" ] || { echo "builder did not finish; inspect $BUILDER"; exit 1; }

gcloud compute images create "$IMAGE_NAME" \
    --project "$PROJECT" \
    --source-disk "$BUILDER" --source-disk-zone "$ZONE" \
    --family "$IMAGE_FAMILY"

gcloud compute instances delete "$BUILDER" --zone "$ZONE" --project "$PROJECT" --quiet

echo "Image $IMAGE_NAME created in family '$IMAGE_FAMILY'."
