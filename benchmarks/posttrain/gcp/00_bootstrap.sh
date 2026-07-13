#!/bin/bash
# One-time GCP project setup for PostTrainBench runs:
# APIs, results bucket, runner service account, API-key secrets.
#
# Prereqs: gcloud authenticated (`gcloud auth login`), billing enabled.
# Secrets are read from your local environment if set:
#   ANTHROPIC_API_KEY (required)  OPENAI_API_KEY (judge tasks)  HF_TOKEN (gated gemma)

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

echo "Project: $PROJECT  Region: $REGION  Zone: $ZONE"

gcloud services enable compute.googleapis.com secretmanager.googleapis.com \
    storage.googleapis.com --project "$PROJECT"

gsutil ls -b "gs://$BUCKET" >/dev/null 2>&1 || \
    gsutil mb -p "$PROJECT" -l "$REGION" -b on "gs://$BUCKET"

gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT" >/dev/null 2>&1 || \
    gcloud iam service-accounts create "$SA_NAME" --project "$PROJECT" \
        --display-name "PostTrainBench runner"

gsutil iam ch "serviceAccount:${SA_EMAIL}:roles/storage.objectAdmin" "gs://$BUCKET"
gcloud projects add-iam-policy-binding "$PROJECT" --quiet \
    --member "serviceAccount:$SA_EMAIL" --role roles/secretmanager.secretAccessor >/dev/null
gcloud projects add-iam-policy-binding "$PROJECT" --quiet \
    --member "serviceAccount:$SA_EMAIL" --role roles/logging.logWriter >/dev/null
# Lets a finished run VM delete itself (early stop => stop paying for the GPU).
# Comment out if too broad for your project; max-run-duration cleans up anyway.
gcloud projects add-iam-policy-binding "$PROJECT" --quiet \
    --member "serviceAccount:$SA_EMAIL" --role roles/compute.instanceAdmin.v1 >/dev/null

upsert_secret() {
    local name="$1" value="${2:-}"
    [ -z "$value" ] && { echo "skip secret $name (env var not set)"; return; }
    if gcloud secrets describe "$name" --project "$PROJECT" >/dev/null 2>&1; then
        printf '%s' "$value" | gcloud secrets versions add "$name" --project "$PROJECT" --data-file=-
    else
        printf '%s' "$value" | gcloud secrets create "$name" --project "$PROJECT" --data-file=-
    fi
}
upsert_secret anthropic-api-key "${ANTHROPIC_API_KEY:-}"
upsert_secret openai-api-key "${OPENAI_API_KEY:-}"
upsert_secret hf-token "${HF_TOKEN:-}"

cat <<EOF

Bootstrap done. REMAINING MANUAL STEP — H100 quota:
  a3-highgpu-1g is only purchasable as Spot or DWS Flex-start, and BOTH consume
  the *preemptible* H100 quota. Request in the console:
    IAM & Admin > Quotas > filter "Preemptible NVIDIA H100 GPUs" > region $REGION > request >= 1
  (metric: compute.googleapis.com/preemptible_nvidia_h100_gpus)
  Small requests are usually auto-approved within minutes-hours.

Also: accept the license for google/gemma-3-4b-pt on huggingface.co with the
account behind HF_TOKEN, or gemma runs cannot download the base model.
EOF
