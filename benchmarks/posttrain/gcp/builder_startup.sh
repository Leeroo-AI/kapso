#!/bin/bash
# Startup script for the asset-builder VM (created by 01_build_assets.sh).
# Builds the containers, uploads them to GCS, fills the HF-cache disk, then
# writes the BUILD_DONE marker and powers off.

set -x
exec > /var/log/ptb-builder.log 2>&1

meta() { curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }

BUCKET=$(meta ptb_bucket)
PTB_REPO=$(meta ptb_repo)
KAPSO_REPO=$(meta kapso_repo)
CACHE_SCOPE=$(meta cache_scope)

finish() {
    gsutil cp /var/log/ptb-builder.log "gs://$BUCKET/assets/build.log" || true
    poweroff
}
trap finish EXIT

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y software-properties-common git rsync jq python3 uuid-runtime
add-apt-repository -y ppa:apptainer/ppa
apt-get update
apt-get install -y apptainer fuse-overlayfs

# --- HF cache disk ---
DISK=/dev/disk/by-id/google-hfcache
blkid "$DISK" || mkfs.ext4 -F -L HFCACHE "$DISK"
mkdir -p /mnt/hfcache
mount "$DISK" /mnt/hfcache
export HF_HOME=/mnt/hfcache/huggingface
mkdir -p "$HF_HOME"

# --- repos ---
git clone --depth 1 "$PTB_REPO" /opt/ptb
# kapso source: prefer the exact local tree uploaded by 01_build_assets.sh
# (kapso_src_gcs metadata) over a git clone, so unpushed branches build too.
KAPSO_SRC_GCS=$(meta kapso_src_gcs || true)
if [ -n "$KAPSO_SRC_GCS" ]; then
    mkdir -p /opt/kapso-src
    gsutil cp "$KAPSO_SRC_GCS" /opt/kapso-src.tgz
    tar -xzf /opt/kapso-src.tgz -C /opt/kapso-src
else
    git clone --depth 1 "$KAPSO_REPO" /opt/kapso-src
fi
cd /opt/ptb

# Apply the kapso adapter (defs always overwritten so def fixes propagate)
ADAPTER=/opt/kapso-src/benchmarks/posttrain/ptb_adapter
if [ -d "$ADAPTER" ]; then
    mkdir -p agents/kapso
    cp "$ADAPTER/agents/kapso/solve.sh" agents/kapso/solve.sh
    cp "$ADAPTER/containers/kapso.def" containers/kapso.def
    cp "$ADAPTER/containers/vllm_debug.def" containers/vllm_debug.def
fi

# --- containers ---
rsync -a --delete --exclude .git --exclude archive --exclude tests \
    --exclude '.env' --exclude '*.env' \
    --exclude build --exclude dist --exclude '*.egg-info' \
    /opt/kapso-src/ containers/kapso-src/
bash containers/build_container.sh kapso
bash containers/build_container.sh vllm_debug

# Fail loudly: no BUILD_DONE (and no snapshot) unless both images exist.
if [ ! -f containers/kapso.sif ] || [ ! -f containers/vllm_debug.sif ]; then
    echo "container build failed" | gsutil cp - "gs://$BUCKET/assets/BUILD_FAILED"
    exit 1
fi

# Smoke-test the kapso entrypoint AND the full runtime import chain (no GPU
# needed) — --help alone missed a missing gated_mcp dependency once.
if ! apptainer exec containers/kapso.sif /opt/kapso/venv/bin/expert-posttrain --help >/dev/null || \
   ! apptainer exec containers/kapso.sif /opt/kapso/venv/bin/python -c \
        "import benchmarks.posttrain.runner, kapso.execution.orchestrator, kapso.execution.search_strategies.generic.strategy, kapso.gated_mcp, kapso.execution.coding_agents.adapters.claude_code_agent"; then
    echo "kapso smoke/import test failed inside container" | gsutil cp - "gs://$BUCKET/assets/BUILD_FAILED"
    exit 1
fi

# Installed-vs-source integrity: a stale build/lib once shipped an outdated
# config.yaml into the venv while the source tree looked correct.
if ! apptainer exec containers/kapso.sif diff -q \
    /opt/kapso-src/benchmarks/posttrain/config.yaml \
    /opt/kapso/venv/lib/python3.10/site-packages/benchmarks/posttrain/config.yaml; then
    echo "installed config.yaml differs from source (stale build artifacts?)" \
        | gsutil cp - "gs://$BUCKET/assets/BUILD_FAILED"
    exit 1
fi

gsutil cp containers/kapso.sif containers/vllm_debug.sif "gs://$BUCKET/assets/"
# Also bake the containers onto the cache disk so run VMs skip the GCS pull.
mkdir -p /mnt/hfcache/containers
cp containers/kapso.sif containers/vllm_debug.sif /mnt/hfcache/containers/

# --- HF cache ---
# Gated models (gemma) need a token whose account accepted the license.
export HF_TOKEN="$(gcloud secrets versions access latest --secret=hf-token 2>/dev/null || true)"

if [ "$CACHE_SCOPE" = "core" ]; then
    cat > containers/download_hf_cache/resources.json <<'EOF'
{
  "models": [
    "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen3-4B-Base",
    "HuggingFaceTB/SmolLM3-3B-Base",
    "google/gemma-3-4b-pt"
  ],
  "datasets": [
    {"dataset": "openai/gsm8k", "configs": ["default", "main"], "splits": ["test", "train"]},
    {"dataset": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
     "configs": ["chatable", "exec_multiple", "exec_parallel_multiple", "exec_simple",
                 "java", "javascript", "parallel", "rest", "sql"],
     "splits": ["train"]}
  ]
}
EOF
fi

# Gated models (google/gemma-3-4b-pt) fail without an accepted-license token;
# keep gemma LAST in the core list and tolerate a partial download.
# Hard timeout + hub socket timeout: a dropped CDN connection once left the
# downloader futex-waiting forever (CLOSE-WAIT sockets, zero progress).
timeout --signal=TERM --kill-after=60s 45m apptainer exec \
    --bind "${HF_HOME}:${HF_HOME}" \
    --bind /opt/ptb:/opt/ptb \
    --env HF_HOME="${HF_HOME}" \
    --env HF_TOKEN="${HF_TOKEN}" \
    --env HF_HUB_DOWNLOAD_TIMEOUT=60 \
    --env HF_HUB_ETAG_TIMEOUT=60 \
    --pwd /opt/ptb \
    containers/kapso.sif python -u containers/download_hf_cache/download_resources.py \
    || echo "WARN: cache download incomplete (timeout/gated model?)"

du -sh "$HF_HOME" | gsutil cp - "gs://$BUCKET/assets/cache_size.txt" || true
umount /mnt/hfcache

echo done | gsutil cp - "gs://$BUCKET/assets/BUILD_DONE"
