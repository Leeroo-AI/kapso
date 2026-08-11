#!/bin/bash
# Kapso agent adapter for PostTrainBench.
#
# Contract (see src/run_task.sh):
#   - cwd is /home/ben/task (evaluate.py, timer.sh, templates/ live here)
#   - $PROMPT holds the official task prompt, $AGENT_CONFIG the model id,
#     $NUM_GPUS the GPU count
#   - CODEX_API_KEY always holds the harness operator's OpenAI key — the
#     sanctioned agent-credential channel (upstream codex scaffolds run on
#     it). OPENAI_API_KEY itself is pre-exported ONLY for judge-scored tasks
#     (arenahardwriting, healthbench), where evaluate.py needs it.
#   - the harness enforces the deadline with `timeout` (TERM, then KILL +30s)
#
# Kapso lives in its own venv (see containers/kapso.def) so its dependencies
# never touch the pinned training environment the agent works in.

# Kapso's utility-LLM roles (gpt-5.6-luna: repo memory, insight extraction —
# see benchmarks/posttrain/config.yaml) authenticate through the CODEX_API_KEY
# channel, bridged to OPENAI_API_KEY for litellm. Whether the harness itself
# pre-exported OPENAI_API_KEY is exactly the judge-task signal: on judge tasks
# the agent sessions must keep the key for evaluate.py; on every other task
# they must look like an official non-judge environment (no OpenAI key), so
# the runner strips it from agent sessions while kapso's own process keeps it.
if [ -n "${OPENAI_API_KEY:-}" ]; then
    STRIP_AGENT_ENV_ARGS=()
else
    export OPENAI_API_KEY="${CODEX_API_KEY:-}"
    STRIP_AGENT_ENV_ARGS=(--strip-agent-env OPENAI_API_KEY)
fi

# The ensemble's codex member authenticates via ~/.codex/auth.json (ChatGPT
# login), never via keys: CODEX_API_KEY would override auth.json if left set.
unset GEMINI_API_KEY
unset CODEX_API_KEY

# Claude Max subscription auth: run_task.sh copies agents/kapso/oauth_token to
# the job home when present (same mechanism as the upstream claude_non_api
# agent). Prefer it over the API key to avoid auth conflicts.
# The file may be MULTI-LINE: line 1 = main token; lines 2+ = recovery tokens
# for the adapter's session-limit failover (a usage-window 429 storm swaps
# the CLI to the next token via --resume; see claude_code_agent.py). The
# recovery list travels in CLAUDE_CODE_OAUTH_RECOVERY_TOKENS (comma-joined);
# the adapter strips it from every agent session's env.
if [ -f /home/ben/oauth_token ]; then
    CLAUDE_CODE_OAUTH_TOKEN="$(head -n 1 /home/ben/oauth_token)"
    export CLAUDE_CODE_OAUTH_TOKEN
    CLAUDE_CODE_OAUTH_RECOVERY_TOKENS="$(tail -n +2 /home/ben/oauth_token | grep -v '^[[:space:]]*$' | paste -sd, -)"
    if [ -n "$CLAUDE_CODE_OAUTH_RECOVERY_TOKENS" ]; then
        export CLAUDE_CODE_OAUTH_RECOVERY_TOKENS
    fi
    unset ANTHROPIC_API_KEY
fi

# Bash-tool clock policy (run #7, finding F5): the DEFAULT timeout is the
# mechanical backstop — an un-annotated blocking call dies after 15 minutes
# with an error that teaches the detached-launch pattern, so no foreground
# call can hold a session hostage. Deliberate long waits stay possible: the
# agent may pass an explicit per-call timeout up to the 10h MAX. Detached
# `nohup … &` launches return instantly and are unaffected.
export BASH_MAX_TIMEOUT_MS="36000000"
export BASH_DEFAULT_TIMEOUT_MS="900000"

# The container runs as root and Claude Code refuses
# --dangerously-skip-permissions under root unless it knows it's sandboxed.
export IS_SANDBOX=1

# HF hub clients hang forever on dropped CDN connections without these
# (observed: futex-wait on CLOSE-WAIT sockets, 15 min of budget lost).
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HUB_ETAG_TIMEOUT=60

# Curated public post-training knowledge bank: copy from the baked source
# tree into the task workspace so the agent reads it as local files (the
# tarball and container are built together, so the bank is always present
# in a matched image — its absence means a stale container).
if [ ! -d /opt/kapso-src/benchmarks/posttrain/knowledge_bank ]; then
    echo "FATAL: knowledge_bank missing from the container image — rebuild assets" >&2
    exit 1
fi
cp -r /opt/kapso-src/benchmarks/posttrain/knowledge_bank "$PWD/knowledge_bank"

exec /opt/kapso/venv/bin/expert-posttrain \
    --task-dir "$PWD" \
    --prompt-env PROMPT \
    --coding-model "${AGENT_CONFIG:-claude-opus-4-6}" \
    --mode POSTTRAIN \
    "${STRIP_AGENT_ENV_ARGS[@]}"
