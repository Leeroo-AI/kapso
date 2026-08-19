#!/bin/bash
# Claude Code baseline for RelBench: one headless Claude Code session per task
# (Fable 5, --effort xhigh), HOURS wall-clock on a 1xA100 GCP box, same
# sanitized cache + starter-kit contract as Kapso, scored one-way afterwards.
#
#   run_baseline.sh box      <box> <zone>                # create 1xA100 box from a lane snapshot
#   run_baseline.sh run      <box> <zone> <ds>/<task> [hours]   # build cache, run session, harvest, score
#   run_baseline.sh harvest  <box> <zone> <ds>/<task>   # re-pull + score an existing run dir
#   run_baseline.sh stop     <box> <zone>
#
# Tasks run SEQUENTIALLY (one session at a time per box). Auth: the
# CLAUDE_CODE_OAUTH_TOKEN3 line of the worktree .env is pushed to the box as
# CLAUDE_CODE_OAUTH_TOKEN.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
RB=$(cd "$HERE/../../.." && pwd)
ENVFILE=$RB/.env
KEY=/home/ubuntu/.ssh/google_compute_engine
PROJECT=gen-lang-client-0664337543
SNAPSHOT_SRC_DISK=relbench-lane-c14
SNAPSHOT_SRC_ZONE=asia-southeast1-c
G="gcloud --project $PROJECT"
SSHOPTS="-i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"
ARCHIVE_BUCKET=gs://leeroo-kapso-relbench-artifacts/baselines/claude_code
PY_REMOTE=/home/ubuntu/miniconda3/bin/python
PY_LOCAL=/home/ubuntu/miniconda3/envs/kapso_conda/bin/python
HARVEST_ROOT=$RB/tmp/claude_code_baseline

CMD=$1; BOX=$2; ZONE=$3

box_ip() { $G compute instances describe "$BOX" --zone "$ZONE" --format='value(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null; }
ssh_box() { ssh $SSHOPTS "ubuntu@$(box_ip)" "$@"; }
wait_ssh() { for i in $(seq 1 40); do ssh_box true 2>/dev/null && return 0; sleep 10; done; echo "SSH-TIMEOUT"; return 1; }

case $CMD in
box)
  SNAP=relbench-lane-snap-$(date -u +%Y%m%d)
  $G compute snapshots describe "$SNAP" >/dev/null 2>&1 || \
    $G compute disks snapshot "$SNAPSHOT_SRC_DISK" --zone "$SNAPSHOT_SRC_ZONE" --snapshot-names "$SNAP" --quiet
  $G compute instances create "$BOX" --zone "$ZONE" --machine-type a2-highgpu-1g \
    --maintenance-policy TERMINATE --source-snapshot "$SNAP" --boot-disk-size 500GB \
    --boot-disk-type pd-balanced --scopes cloud-platform --quiet
  wait_ssh && ssh_box 'nvidia-smi -L; ls ~/miniconda3/bin/python ~/kapso >/dev/null && echo BOX-READY'
  ;;

run|harvest)
  TASK=$4; HOURS=${5:-4}
  DS=${TASK%%/*}; TN=${TASK##*/}; DIR=${TASK/\//--}
  IP=$(box_ip); [ -z "$IP" ] && { echo "box not running"; exit 1; }
  STAMP=$(date -u +%Y%m%dT%H%M%S)
  if [ "$CMD" = run ]; then
    # --- auth: TOKEN3 -> box as CLAUDE_CODE_OAUTH_TOKEN
    T=$(grep -E '^CLAUDE_CODE_OAUTH_TOKEN3=' "$ENVFILE" | cut -d= -f2- | tr -d '"'"'"' \r')
    [ ${#T} -lt 40 ] && { echo "TOKEN3 unreadable"; exit 1; }
    ssh_box "mkdir -p ~/cc_baseline && echo 'CLAUDE_CODE_OAUTH_TOKEN=$T' > ~/cc_baseline/.env"
    # --- code: repo HEAD (sandbox builder + starter kit) to the box
    STAGE=$(mktemp -d); git -C "$RB" archive HEAD benchmarks/relbench src | tar -x -C "$STAGE"
    rsync -az -e "ssh $SSHOPTS" "$STAGE/benchmarks/" "ubuntu@$IP:kapso/benchmarks/"
    rsync -az -e "ssh $SSHOPTS" "$STAGE/src/" "ubuntu@$IP:kapso/src/"
    rm -rf "$STAGE"
    # --- prompt
    sed -e "s#{dataset}#$DS#g" -e "s#{task}#$TN#g" -e "s#{hours}#$HOURS#g" \
        -e "s#{hardware}#1x A100 40GB, 12 vCPU, 85 GB RAM#g" "$HERE/PROMPT.md" > /tmp/cc_prompt_$DIR.md
    scp $SSHOPTS -q /tmp/cc_prompt_$DIR.md "ubuntu@$IP:cc_baseline/prompt_$DIR.md"
    # --- sanitized cache + fresh workdir with the starter kit, then the session
    ssh_box "set -e
      W=~/cc_baseline/$DIR/$STAMP; mkdir -p \$W; cd ~/kapso
      if [ ! -d ~/cc_baseline/cache_$DIR/$DS/tasks/$TN ]; then
        $PY_REMOTE -c 'from relbench.datasets import get_dataset; from relbench.tasks import get_task
get_dataset(\"$DS\", download=True).get_db(); get_task(\"$DS\", \"$TN\", download=True)
print(\"pristine cache ready\")' > \$W/download.log 2>&1
        PYTHONPATH=src:. $PY_REMOTE -m benchmarks.relbench.sandbox --dataset $DS --task $TN \
          --dest ~/cc_baseline/cache_$DIR --source ~/.cache/relbench > \$W/sandbox.log 2>&1
        test -d ~/cc_baseline/cache_$DIR/$DS/tasks/$TN || { echo SANDBOX-FAILED; tail -3 \$W/sandbox.log; exit 1; }
      fi
      cp -r ~/kapso/benchmarks/relbench/data/starter_kit \$W/kapso_datasets
      rm -rf \$W/kapso_datasets/__pycache__; mkdir -p \$W/kapso_output
      cat > \$W/session.sh <<EOF
set -a; . ~/cc_baseline/.env; set +a
export RELBENCH_CACHE_DIR=\$HOME/cc_baseline/cache_$DIR RELBENCH_DATASET=$DS RELBENCH_TASK=$TN
export KAPSO_RUN_DATA_DIR=\$W/kapso_output KAPSO_SHARED_CACHE_DIR=\$W/shared_cache
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=\$W/shared_cache/hf
export PATH=\$HOME/miniconda3/bin:\$PATH
cd \$W
date -u +%FT%TZ > started_at
timeout ${HOURS}h claude -p --dangerously-skip-permissions --model claude-fable-5 --effort xhigh \
  --output-format stream-json --verbose < ~/cc_baseline/prompt_$DIR.md > session.jsonl 2> session.err
echo EXIT=\\\$? > exit_code; date -u +%FT%TZ > ended_at
EOF
      tmux new-session -d -s cc_$DIR 'bash '\$W'/session.sh'
      echo \$W > ~/cc_baseline/current_$DIR
      echo LAUNCHED \$W"
    echo "[$BOX] $TASK launched; waiting ${HOURS}h + grace"
    # --- wait for the session to end (exit_code file appears)
    until ssh_box "test -f \$(cat ~/cc_baseline/current_$DIR)/exit_code" 2>/dev/null; do sleep 300; done
  fi
  # --- harvest: pull the run dir (minus caches), score, archive
  W=$(ssh_box "cat ~/cc_baseline/current_$DIR")
  LOCAL=$HARVEST_ROOT/$DIR/$(basename "$W"); mkdir -p "$LOCAL"
  rsync -az --exclude shared_cache -e "ssh $SSHOPTS" "ubuntu@$IP:$W/" "$LOCAL/"
  $PY_LOCAL "$HERE/score_baseline.py" "$TASK" "$LOCAL" | tee -a "$LOCAL/score.txt"
  tar czf "$LOCAL.tgz" -C "$HARVEST_ROOT/$DIR" "$(basename "$W")" && \
    gsutil -q cp "$LOCAL.tgz" "$ARCHIVE_BUCKET/$DIR/$(basename "$W").tgz" && echo "archived -> $ARCHIVE_BUCKET/$DIR/$(basename "$W").tgz"
  ;;

stop)
  $G compute instances stop "$BOX" --zone "$ZONE" --quiet
  ;;
*) echo "usage: $0 box|run|harvest|stop <box> <zone> [<ds>/<task> [hours]]"; exit 1 ;;
esac
