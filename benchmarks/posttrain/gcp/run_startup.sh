#!/bin/bash
# Startup script for a PostTrainBench run VM (created by 10_launch_run.sh).
# Installs driver+apptainer, mounts the HF-cache disk clone, pulls the
# prebuilt containers, runs src/run_task.sh, streams results to GCS, and
# deletes the VM when finished.

set -x
exec > /var/log/ptb-run.log 2>&1

meta() { curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }

EVAL=$(meta ptb_eval)
MODEL=$(meta ptb_model)
HOURS=$(meta ptb_hours)
AGENT_CONFIG=$(meta ptb_agent_config)
BUCKET=$(meta ptb_bucket)
PTB_REPO=$(meta ptb_repo)
KAPSO_REPO=$(meta kapso_repo)
RUN_ID=$(meta ptb_run_id)

RESULTS_GS="gs://$BUCKET/results/$RUN_ID"

self_destruct() {
    code=$?
    gsutil cp /var/log/ptb-run.log "$RESULTS_GS/ptb-run.log" || true
    [ -d /opt/ptb/results ] && gsutil -m rsync -r /opt/ptb/results "$RESULTS_GS/results" || true
    echo "exit_code=$code" | gsutil cp - "$RESULTS_GS/RUN_DONE" || true
    NAME=$(curl -sf -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/name)
    VMZONE=$(curl -sf -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
    gcloud compute instances delete "$NAME" --zone "$VMZONE" --quiet || poweroff
}
trap self_destruct EXIT

# Skipped when booting from the golden image (02_build_image.sh).
if [ ! -f /etc/ptb-image-ready ]; then
    export DEBIAN_FRONTEND=noninteractive
    dpkg --configure -a || true   # self-heal a dpkg interrupted at image time
    apt-get update
    apt-get install -y software-properties-common git rsync jq python3 uuid-runtime tree mdadm
    add-apt-repository -y ppa:apptainer/ppa
    apt-get update
    apt-get install -y apptainer fuse-overlayfs
    apt-get install -y nvidia-driver-570-server || apt-get install -y nvidia-driver-550-server
fi
for _ in $(seq 1 40); do nvidia-smi && break; sleep 15; done
nvidia-smi || exit 1

# --- local SSD (bundled with a3-highgpu-1g) -> fast /tmp for job dirs ---
mapfile -t SSDS < <(ls /dev/disk/by-id/google-local-nvme-ssd-* 2>/dev/null | grep -v part || true)
if [ "${#SSDS[@]}" -ge 1 ]; then
    if [ "${#SSDS[@]}" -ge 2 ]; then
        mdadm --create /dev/md0 --level=0 --force --raid-devices="${#SSDS[@]}" "${SSDS[@]}"
        TMPDEV=/dev/md0
    else
        TMPDEV="${SSDS[0]}"
    fi
    mkfs.ext4 -F "$TMPDEV"
    mkdir -p /mnt/localssd
    mount "$TMPDEV" /mnt/localssd
    mkdir -p /mnt/localssd/tmp
    chmod 1777 /mnt/localssd/tmp
    mount --bind /mnt/localssd/tmp /tmp   # run_task.sh hardcodes /tmp for job dirs
fi

# --- HF cache disk (clone of the snapshot; agent writes go to an overlay) ---
mkdir -p /mnt/hfcache
mount /dev/disk/by-id/google-hfcache /mnt/hfcache
export HF_HOME=/mnt/hfcache/huggingface

# --- secrets ---
export ANTHROPIC_API_KEY="$(gcloud secrets versions access latest --secret=anthropic-api-key 2>/dev/null || true)"
CLAUDE_OAUTH="$(gcloud secrets versions access latest --secret=claude-oauth-token 2>/dev/null || true)"
if [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$CLAUDE_OAUTH" ]; then
    echo "FATAL: need claude-oauth-token or anthropic-api-key secret"; exit 1
fi
export OPENAI_API_KEY="$(gcloud secrets versions access latest --secret=openai-api-key 2>/dev/null || true)"
HF_TOKEN="$(gcloud secrets versions access latest --secret=hf-token 2>/dev/null || true)"
# huggingface_hub reads $HF_HOME/token — makes gated models work in-container.
[ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > "$HF_HOME/token"

# --- PostTrainBench checkout + kapso adapter + containers ---
git clone --depth 1 "$PTB_REPO" /opt/ptb
cd /opt/ptb
if [ ! -f agents/kapso/solve.sh ]; then
    git clone --depth 1 "$KAPSO_REPO" /opt/kapso-src
    mkdir -p agents/kapso
    cp /opt/kapso-src/benchmarks/posttrain/ptb_adapter/agents/kapso/solve.sh agents/kapso/solve.sh
    cp /opt/kapso-src/benchmarks/posttrain/ptb_adapter/containers/kapso.def containers/kapso.def
fi
# Claude Max subscription: run_task.sh copies this file into the job home and
# solve.sh exports it as CLAUDE_CODE_OAUTH_TOKEN.
[ -n "$CLAUDE_OAUTH" ] && printf '%s' "$CLAUDE_OAUTH" > agents/kapso/oauth_token
# Containers: prefer the copies baked onto the cache-disk snapshot (zero
# download); fall back to GCS (~2-3 min at the ~150 MiB/s we measured).
if [ -f /mnt/hfcache/containers/kapso.sif ]; then
    export POST_TRAIN_BENCH_CONTAINERS_DIR=/mnt/hfcache/containers
else
    gsutil cp "gs://$BUCKET/assets/kapso.sif" "gs://$BUCKET/assets/vllm_debug.sif" containers/
    export POST_TRAIN_BENCH_CONTAINERS_DIR=containers
fi

export POST_TRAIN_BENCH_CONTAINER_NAME=kapso
export POST_TRAIN_BENCH_RESULTS_DIR=results
export POST_TRAIN_BENCH_JOB_SCHEDULER=local

# --- crash-safe periodic results upload ---
# Measured on spot: preemption grace is 29s and may be 0s mid-boot, and the
# job dir sits on ephemeral local SSD — only what's already synced survives.
( while sleep 300; do
      gsutil -m rsync -r results "$RESULTS_GS/results" >/dev/null 2>&1 || true
  done ) &
UPLOADER=$!

bash src/run_task.sh "$EVAL" kapso "$MODEL" "$RUN_ID" "$HOURS" "$AGENT_CONFIG" 1

kill "$UPLOADER" 2>/dev/null || true
# final sync happens in self_destruct
