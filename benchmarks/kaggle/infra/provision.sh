#!/usr/bin/env bash
# =============================================================================
# provision.sh — (dev side) create the GPU box for a competition run.
#
# Defaults to an 8xL4 g2 box. Boots from the golden image if IMAGE_FAMILY is
# set (fast, ready), else from the base DLVM (then run setup_box.sh on it once).
# One image serves any GPU count — the count is the machine type:
#   g2-standard-96=8xL4  g2-standard-48=4xL4  g2-standard-24=2xL4  g2-standard-8=1xL4
# =============================================================================
set -euo pipefail
PROJECT="${PROJECT:-trans-density-437811-p2}"
ZONE="${ZONE:-us-central1-b}"                 # L4 zone (us-central1-a hit stockout; -b worked)
NAME="${NAME:-kaggle-8xl4}"
MACHINE="${MACHINE:-g2-standard-96}"          # 8xL4
DISK_GB="${DISK_GB:-300}"
IMAGE_FAMILY="${IMAGE_FAMILY:-}"              # set once a golden image is baked
BASE_IMAGE_FAMILY="${BASE_IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
BASE_IMAGE_PROJECT="${BASE_IMAGE_PROJECT:-deeplearning-platform-release}"

if [ -n "$IMAGE_FAMILY" ]; then
  img=(--image-family "$IMAGE_FAMILY" --image-project "$PROJECT")
else
  img=(--image-family "$BASE_IMAGE_FAMILY" --image-project "$BASE_IMAGE_PROJECT")
fi

gcloud compute instances create "$NAME" \
  --project "$PROJECT" --zone "$ZONE" \
  --machine-type "$MACHINE" \
  --maintenance-policy TERMINATE \
  --boot-disk-size "${DISK_GB}GB" --boot-disk-type pd-ssd \
  "${img[@]}" \
  --metadata install-nvidia-driver=True

echo
echo "created: $NAME  ($MACHINE, $ZONE)"
echo "ssh:     gcloud compute ssh $NAME --zone $ZONE --project $PROJECT"
echo "delete:  gcloud compute instances delete $NAME --zone $ZONE --project $PROJECT -q"
