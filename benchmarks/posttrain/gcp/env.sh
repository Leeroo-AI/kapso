# Shared settings for the PostTrainBench-on-GCP scripts. Source before use or
# let the scripts source it. Override anything via the environment.

export PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
# a3-highgpu-1g exists in ~16 regions; pick one where you hold (or request)
# quota. Check zones with:
#   gcloud compute machine-types list --filter="name=a3-highgpu-1g" --format="value(zone)"
export REGION="${REGION:-us-central1}"
export ZONE="${ZONE:-us-central1-a}"

export BUCKET="${BUCKET:-${PROJECT}-posttrainbench}"
export SA_NAME="${SA_NAME:-ptb-runner}"
export SA_EMAIL="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"

# Fork of https://github.com/aisa-group/PostTrainBench with the kapso adapter
# applied (agents/kapso + containers/kapso.def). Must be readable by the VMs
# (public fork, or grant access another way).
export PTB_REPO_URL="${PTB_REPO_URL:-https://github.com/aisa-group/PostTrainBench.git}"
export KAPSO_REPO_URL="${KAPSO_REPO_URL:-https://github.com/leeroo-ai/kapso.git}"

export CACHE_DISK="${CACHE_DISK:-ptb-hf-cache}"
export CACHE_SNAPSHOT="${CACHE_SNAPSHOT:-ptb-hf-cache-snap}"
# 'core' = 4 base models only (~50 GB, minutes). 'full' = everything in
# resources.json (hundreds of GB, hours) for parity with official runs.
export CACHE_SCOPE="${CACHE_SCOPE:-core}"
export CACHE_DISK_SIZE_GB="${CACHE_DISK_SIZE_GB:-500}"   # use >=1500 for CACHE_SCOPE=full
