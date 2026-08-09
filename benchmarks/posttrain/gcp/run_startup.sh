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
PTB_PIN=$(meta ptb_pin)
KAPSO_REPO=$(meta kapso_repo)
RUN_ID=$(meta ptb_run_id)

RESULTS_GS="gs://$BUCKET/results/$RUN_ID"

RUN_EXIT="startup-died-early"
self_destruct() {
    code=$?
    gsutil cp /var/log/ptb-run.log "$RESULTS_GS/ptb-run.log" || true
    [ -d /opt/ptb/results ] && gsutil -m rsync -r /opt/ptb/results "$RESULTS_GS/results" || true
    echo "exit_code=$code run_task_exit=$RUN_EXIT" | gsutil cp - "$RESULTS_GS/RUN_DONE" || true
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
# run_task.sh calls bare `python` on the host (prompt/judge/trace helpers);
# Ubuntu ships only python3. A transient apt failure here self-destructed a
# run at preflight (gemma 07241426): retry once, then fall back to a manual
# symlink — python3 is guaranteed above, so `python` must never be the sole
# reason a $65 boot dies. Preflight remains the fail-loud backstop.
command -v python >/dev/null || apt-get install -y python-is-python3 \
    || { apt-get update && apt-get install -y python-is-python3; } || true
command -v python >/dev/null || ln -sf "$(command -v python3)" /usr/local/bin/python

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

# --- secrets (xtrace off: values must never reach the log) ---
set +x
export ANTHROPIC_API_KEY="$(gcloud secrets versions access latest --secret=anthropic-api-key 2>/dev/null || true)"
CLAUDE_OAUTH="$(gcloud secrets versions access latest --secret=claude-oauth-token 2>/dev/null || true)"
if [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$CLAUDE_OAUTH" ]; then
    echo "FATAL: need claude-oauth-token or anthropic-api-key secret"; set -x; exit 1
fi
export OPENAI_API_KEY="$(gcloud secrets versions access latest --secret=openai-api-key 2>/dev/null || true)"
HF_TOKEN="$(gcloud secrets versions access latest --secret=hf-token 2>/dev/null || true)"
# huggingface_hub reads $HF_HOME/token — makes gated models work in-container.
[ -n "$HF_TOKEN" ] && printf '%s' "$HF_TOKEN" > "$HF_HOME/token"
echo "secrets loaded: anthropic=$([ -n "$ANTHROPIC_API_KEY" ] && echo yes || echo no) oauth=$([ -n "$CLAUDE_OAUTH" ] && echo yes || echo no) openai=$([ -n "$OPENAI_API_KEY" ] && echo yes || echo no) hf=$([ -n "$HF_TOKEN" ] && echo yes || echo no)"
set -x

# --- PostTrainBench checkout + kapso adapter + containers ---
# v1.1 evaluator (four judges: data_contamination / api_usage / ptb_lookup /
# general) pinned to a fixed commit ($PTB_PIN, from env.sh PTB_PIN_COMMIT) so
# the evaluator can't drift or vanish when the new_judge_v2 branch moves or
# merges. Fetch-by-SHA works on GitHub even after the source branch is deleted.
git init -q /opt/ptb
git -C /opt/ptb remote add origin "$PTB_REPO"
git -C /opt/ptb fetch --depth 1 origin "$PTB_PIN"
git -C /opt/ptb checkout -q "$PTB_PIN"
cd /opt/ptb
# Adapter comes from the same tarball the container was built from — never a
# git branch (a clone of the default branch once shipped a checkout with no
# agents/kapso/solve.sh and the run burned GPU time on a nonexistent agent).
if [ ! -f agents/kapso/solve.sh ]; then
    gsutil cp "gs://$BUCKET/assets/kapso-src.tgz" /opt/kapso-src.tgz
    mkdir -p /opt/kapso-src
    tar -xzf /opt/kapso-src.tgz -C /opt/kapso-src
    mkdir -p agents/kapso
    cp /opt/kapso-src/benchmarks/posttrain/ptb_adapter/agents/kapso/solve.sh agents/kapso/solve.sh
fi
# v1.1 API-key allowlist: run_task.sh launches the agent sandbox with
# `-c --cleanenv`, so it inherits NOTHING from the host and injects ONLY the
# keys in agents/kapso/api_keys.json (unioned with the task's required_api_keys)
# via --env. The pure-CLI agents (codex/claude) declare `[]` — they authenticate
# solely through bind-mounted auth.json/oauth_token. kapso differs: its ensemble
# also runs litellm utility/reasoning/embedding roles that need the operator's
# OpenAI key, delivered through the sanctioned CODEX_API_KEY channel (run_task.sh
# sets CODEX_API_KEY=$OPENAI_API_KEY then blanks OPENAI_API_KEY on non-judge
# tasks; solve.sh bridges it back to OPENAI_API_KEY for litellm and strips it
# from the agent sessions). Allowing `[]` here would starve kapso's litellm with
# an empty key; CODEX_API_KEY must pass through.
mkdir -p agents/kapso
printf '{"allowed_api_keys": ["CODEX_API_KEY"]}\n' > agents/kapso/api_keys.json
# The v1.1 judges default to a gpt_5_5.sif container, but its upstream def has
# an unsatisfiable vllm==0.11.0/xformers pin (build fails). The judges need
# only codex + node + python — all present in kapso.sif (the agent container,
# which installs @openai/codex) — so repoint JUDGE_CONTAINER there instead.
sed -i 's/JUDGE_CONTAINER="gpt_5_5.sif"/JUDGE_CONTAINER="kapso.sif"/' src/judges/judge_lib.sh
# v1.1 harness: run_task.sh sources src/commit_utils/set_env_vars.sh, which
# hard-requires /opt/ptb/.env to exist (the old harness didn't). It exports
# .env vars but NEVER overrides an already-set env var, and run_startup.sh
# already exports the real keys/paths/config — so a comment-only .env
# satisfies the check without clobbering any of our exports.
printf '# kapso run: harness env is provided by run_startup.sh exports.\n' > /opt/ptb/.env
# v1.1 judges pin their own models via src/judges/<judge>/judge.conf
# (gpt-5.4 for contamination/api/lookup; gpt-5.6-terra + codex 0.144.5 for
# general) — there is no gpt-5.1-codex to re-pin on this branch. They run via
# codex INSIDE the gpt_5_5.sif judge container and read ChatGPT-subscription
# auth from agents/codex_non_api/auth.json (written below).
# Claude Max subscription: run_task.sh copies this file into the job home and
# solve.sh exports it as CLAUDE_CODE_OAUTH_TOKEN. Codex ChatGPT login: the
# harness copies agents/<agent>/auth.json to the job's ~/.codex/auth.json —
# the ensemble's codex member authenticates through it, never through the
# harness's OPENAI_API_KEY. (xtrace off: secret values)
set +x
[ -n "$CLAUDE_OAUTH" ] && printf '%s' "$CLAUDE_OAUTH" > agents/kapso/oauth_token
# Recovery tokens (session-limit failover): the claude-oauth-recovery secret
# holds newline-separated spare tokens. They ride as EXTRA LINES of the same
# oauth_token file (run_task.sh copies only that one file); solve.sh splits
# line 1 = main, lines 2+ = recovery.
CLAUDE_RECOVERY="$(gcloud secrets versions access latest --secret=claude-oauth-recovery 2>/dev/null || true)"
if [ -n "$CLAUDE_OAUTH" ] && [ -n "$CLAUDE_RECOVERY" ]; then
    printf '\n%s' "$CLAUDE_RECOVERY" >> agents/kapso/oauth_token
fi
echo "oauth recovery tokens present: $([ -n "$CLAUDE_RECOVERY" ] && echo yes || echo no)"
CODEX_AUTH="$(gcloud secrets versions access latest --secret=codex-auth-json 2>/dev/null || true)"
# Two consumers of the same ChatGPT-subscription codex auth:
#   agents/kapso/auth.json         -> the agent ensemble's codex member (run phase)
#   agents/codex_non_api/auth.json -> the v1.1 judges (judge_lib.sh JUDGE_CODEX_AUTH_SRC)
if [ -n "$CODEX_AUTH" ]; then
    printf '%s' "$CODEX_AUTH" > agents/kapso/auth.json
    mkdir -p agents/codex_non_api
    printf '%s' "$CODEX_AUTH" > agents/codex_non_api/auth.json
fi
echo "codex auth present: $([ -n "$CODEX_AUTH" ] && echo yes || echo no)"
set -x
# Containers: prefer the copies baked onto the cache-disk snapshot (zero
# download); fall back to GCS (~2-3 min at the ~150 MiB/s we measured).
if [ -f /mnt/hfcache/containers/kapso.sif ]; then
    export POST_TRAIN_BENCH_CONTAINERS_DIR=/mnt/hfcache/containers
else
    gsutil cp "gs://$BUCKET/assets/kapso.sif" "gs://$BUCKET/assets/vllm_debug.sif" containers/
    export POST_TRAIN_BENCH_CONTAINERS_DIR=containers
fi

export POST_TRAIN_BENCH_CONTAINER_NAME=kapso
# MUST be absolute: the final evaluation resolves --model-path from a
# different cwd (src/eval/tasks/<task>), so a relative results dir makes
# vLLM treat "results/.../final_model" as a hub repo id and die on startup.
# This killed the official eval in every run until #7 produced a model.
export POST_TRAIN_BENCH_RESULTS_DIR=/opt/ptb/results
export POST_TRAIN_BENCH_JOB_SCHEDULER=local

# --- pre-flight: fail before the GPU phase, not during it ---
PREFLIGHT=""
[ -f agents/kapso/solve.sh ] || PREFLIGHT="$PREFLIGHT solve.sh"
[ -f "$POST_TRAIN_BENCH_CONTAINERS_DIR/kapso.sif" ] || PREFLIGHT="$PREFLIGHT kapso.sif"
[ -f "$POST_TRAIN_BENCH_CONTAINERS_DIR/vllm_debug.sif" ] || PREFLIGHT="$PREFLIGHT vllm_debug.sif"
[ -d "$HF_HOME/hub" ] || PREFLIGHT="$PREFLIGHT hf-cache"
command -v python >/dev/null || PREFLIGHT="$PREFLIGHT host-python"
if [ -n "$PREFLIGHT" ]; then
    echo "PREFLIGHT FAILED:$PREFLIGHT"
    RUN_EXIT=preflight
    exit 1
fi

# --- v1.1 self-decontamination test set ---
# The v1.1 harness hard-requires src/eval/tasks/<task>/test_data.json: the
# canonical test set that both the agent's own contamination_check.py and the
# data-contamination judge screen training data against. It is NOT committed to
# git — src/judges/test_data_download/download_test_data.py fetches and
# normalizes it from the benchmark's upstream source. Run it inside kapso.sif
# (which carries huggingface_hub/datasets) for just this run's task, binding the
# host /opt/ptb so the JSON lands in the tree run_task.sh reads. MY_HF_TOKEN is
# consumed only by the gated GPQA downloader; harmless for other tasks.
export MY_HF_TOKEN="$HF_TOKEN"
apptainer exec --bind /opt/ptb:/opt/ptb "$POST_TRAIN_BENCH_CONTAINERS_DIR/kapso.sif" \
    python /opt/ptb/src/judges/test_data_download/download_test_data.py --tasks "$EVAL"
if [ ! -f "src/eval/tasks/$EVAL/test_data.json" ]; then
    echo "PREFLIGHT FAILED: test_data.json ($EVAL) not produced by download_test_data.py"
    RUN_EXIT=testdata
    exit 1
fi
echo "test_data.json ready for $EVAL: $(wc -c < "src/eval/tasks/$EVAL/test_data.json") bytes"

# --- crash-safe periodic results upload ---
# Measured on spot: preemption grace is 29s and may be 0s mid-boot, and the
# job dir sits on ephemeral local SSD — only what's already synced survives.
( while sleep 300; do
      gsutil -m rsync -r results "$RESULTS_GS/results" >/dev/null 2>&1 || true
  done ) &
UPLOADER=$!

bash src/run_task.sh "$EVAL" kapso "$MODEL" "$RUN_ID" "$HOURS" "$AGENT_CONFIG" 1
RUN_EXIT=$?

kill "$UPLOADER" 2>/dev/null || true
# final sync happens in self_destruct
