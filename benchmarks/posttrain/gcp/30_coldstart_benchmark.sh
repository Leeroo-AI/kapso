#!/bin/bash
# Cold-start benchmark: measure submit -> "agent could start working" on the
# exact run-VM stack (golden image if present, cache-disk clone, containers,
# kapso venv), without launching a paid 10h agent run.
#
# Phases are timed inside the VM (relative to startup-script start), written
# to guest attributes AND uploaded to gs://$BUCKET/assets/coldstart_<vm>.txt;
# the VM then deletes itself. Cost: ~10 min of H100 flex-start (~$1).
#
# Usage: bash 30_coldstart_benchmark.sh [--spot]

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

SPOT=0
[ "${1:-}" = "--spot" ] && SPOT=1

VM="ptb-coldstart-$(date +%H%M%S)"
DISK="ptb-cache-$VM"

STARTUP=$(mktemp)
cat > "$STARTUP" <<'EOS'
#!/bin/bash
set -x
exec > /var/log/ptb-coldstart.log 2>&1
T0=$(date +%s.%N)
GA="http://metadata.google.internal/computeMetadata/v1/instance/guest-attributes/cs"
meta() { curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
mark() {
    V=$(echo "$(date +%s.%N) $T0" | awk '{printf "%.1f", $1-$2}')
    echo "$1=$V" >> /var/tmp/cs.txt
    curl -sf -X PUT --data "$V" -H "Metadata-Flavor: Google" "$GA/$1" || true
}
put() { echo "$1=$2" >> /var/tmp/cs.txt; curl -sf -X PUT --data "$2" -H "Metadata-Flavor: Google" "$GA/$1" || true; }

BUCKET=$(meta ptb_bucket)
PTB_REPO=$(meta ptb_repo)

finish() {
    gsutil cp /var/tmp/cs.txt "gs://$BUCKET/assets/coldstart_$(hostname).txt" || true
    gsutil cp /var/log/ptb-coldstart.log "gs://$BUCKET/assets/coldstart_$(hostname).log" || true
    NAME=$(curl -sf -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/name)
    VMZONE=$(curl -sf -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
    gcloud compute instances delete "$NAME" --zone "$VMZONE" --quiet || poweroff
}
trap finish EXIT

put image_prebaked "$([ -f /etc/ptb-image-ready ] && echo yes || echo no)"
if [ ! -f /etc/ptb-image-ready ]; then
    export DEBIAN_FRONTEND=noninteractive
    dpkg --configure -a || true   # self-heal a dpkg interrupted at image time
    apt-get update
    apt-get install -y software-properties-common git rsync jq python3 uuid-runtime tree mdadm
    add-apt-repository -y ppa:apptainer/ppa
    apt-get update
    apt-get install -y apptainer fuse-overlayfs
    apt-get install -y nvidia-driver-570-server || apt-get install -y nvidia-driver-550-server
    mark packages_installed
fi

for _ in $(seq 1 60); do nvidia-smi && break; sleep 5; done
nvidia-smi || { put FAILED driver; exit 1; }
mark driver_ready

mapfile -t SSDS < <(ls /dev/disk/by-id/google-local-nvme-ssd-* 2>/dev/null | grep -v part || true)
if [ "${#SSDS[@]}" -ge 2 ]; then
    mdadm --create /dev/md0 --level=0 --force --raid-devices="${#SSDS[@]}" "${SSDS[@]}"
    mkfs.ext4 -F /dev/md0 && mkdir -p /mnt/localssd && mount /dev/md0 /mnt/localssd
    mkdir -p /mnt/localssd/tmp && chmod 1777 /mnt/localssd/tmp && mount --bind /mnt/localssd/tmp /tmp
fi
mark localssd_tmp_ready

mkdir -p /mnt/hfcache
mount /dev/disk/by-id/google-hfcache /mnt/hfcache
mark cache_disk_mounted

TS=$(date +%s.%N)
gcloud secrets versions access latest --secret=hf-token >/dev/null 2>&1 || true
put secret_fetch_secs "$(echo "$(date +%s.%N) $TS" | awk '{printf "%.1f", $1-$2}')"

git clone --depth 1 "$PTB_REPO" /opt/ptb
mark ptb_repo_cloned

SIF=/mnt/hfcache/containers/kapso.sif
if [ ! -f "$SIF" ]; then
    gsutil cp "gs://$BUCKET/assets/kapso.sif" /opt/kapso.sif
    SIF=/opt/kapso.sif
    put sif_source gcs
else
    put sif_source cache_disk
fi
mark containers_ready

apptainer exec --nv "$SIF" nvidia-smi && mark apptainer_gpu_ok
apptainer exec "$SIF" python -c "import torch, transformers, trl, peft" && mark container_imports_ok
apptainer exec "$SIF" python -c "import vllm" && mark vllm_import_ok
apptainer exec "$SIF" /opt/kapso/venv/bin/expert-posttrain --help >/dev/null && mark kapso_cli_ok

# Snapshot lazy-hydration cost: first-touch vs cached read of model weights
F=$(find /mnt/hfcache/huggingface -name "*.safetensors" -path "*Qwen3-1.7B*" 2>/dev/null | head -1)
if [ -n "$F" ]; then
    put model_file_gb "$(stat -c%s "$F" | awk '{printf "%.1f", $1/1e9}')"
    TA=$(date +%s.%N); cat "$F" > /dev/null; TB=$(date +%s.%N); cat "$F" > /dev/null; TC=$(date +%s.%N)
    put weights_first_read_secs "$(echo "$TB $TA" | awk '{printf "%.1f", $1-$2}')"
    put weights_cached_read_secs "$(echo "$TC $TB" | awk '{printf "%.1f", $1-$2}')"
else
    put weights_first_read_secs "NO_MODEL_IN_CACHE"
fi

mark total_agent_ready
EOS

if gcloud compute images describe-from-family ptb-runner --project "$PROJECT" >/dev/null 2>&1; then
    IMAGE_ARGS=(--image-family ptb-runner --image-project "$PROJECT")
else
    IMAGE_ARGS=(--image-family ubuntu-2204-lts --image-project ubuntu-os-cloud)
fi
if [ "$SPOT" = 1 ]; then
    PROVISIONING=(--provisioning-model=SPOT)
else
    PROVISIONING=(--provisioning-model=FLEX_START --request-valid-for-duration=1h)
fi

T0=$(date +%s)
gcloud compute disks create "$DISK" --project "$PROJECT" --zone "$ZONE" \
    --source-snapshot "$CACHE_SNAPSHOT" --type pd-balanced
T_DISK=$(date +%s)
echo "disk_from_snapshot_secs=$((T_DISK - T0))"

gcloud compute instances create "$VM" \
    --project "$PROJECT" --zone "$ZONE" \
    --machine-type a3-highgpu-1g \
    "${PROVISIONING[@]}" \
    --max-run-duration=30m \
    --instance-termination-action=DELETE \
    --maintenance-policy=TERMINATE \
    --reservation-affinity=none \
    --network-interface nic-type=GVNIC \
    "${IMAGE_ARGS[@]}" \
    --boot-disk-size 100GB --boot-disk-type pd-ssd \
    --disk "name=${DISK},device-name=hfcache,mode=rw,auto-delete=yes" \
    --service-account "$SA_EMAIL" --scopes cloud-platform \
    --metadata-from-file startup-script="$STARTUP" \
    --metadata "ptb_bucket=${BUCKET},ptb_repo=${PTB_REPO_URL},enable-guest-attributes=TRUE" \
    --async
rm -f "$STARTUP"
T_SUBMIT=$(date +%s)

for _ in $(seq 1 90); do
    S=$(gcloud compute instances describe "$VM" --zone "$ZONE" --project "$PROJECT" \
        --format="value(status)" 2>/dev/null || echo PENDING)
    [ "$S" = "RUNNING" ] && break
    sleep 5
done
echo "submit_to_running_secs=$(( $(date +%s) - T_SUBMIT ))"

RESULT="gs://$BUCKET/assets/coldstart_${VM}.txt"
echo "Waiting for probe results at $RESULT ..."
for _ in $(seq 1 150); do
    if gsutil -q stat "$RESULT" 2>/dev/null; then
        echo; echo "=== cold-start phases (secs since VM startup-script start) ==="
        gsutil cat "$RESULT"
        echo "total_submit_to_ready_secs=$(( $(date +%s) - T_SUBMIT ))  (approx, incl. boot)"
        exit 0
    fi
    sleep 10
done
echo "Timed out waiting for results; check: gcloud compute instances tail-serial-port-output $VM --zone $ZONE"
exit 1
