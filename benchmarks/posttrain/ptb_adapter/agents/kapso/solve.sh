#!/bin/bash
# Kapso agent adapter for PostTrainBench.
#
# Contract (see src/run_task.sh):
#   - cwd is /home/ben/task (evaluate.py, timer.sh, templates/ live here)
#   - $PROMPT holds the official task prompt, $AGENT_CONFIG the model id,
#     $NUM_GPUS the GPU count
#   - ANTHROPIC_API_KEY is always exported; OPENAI_API_KEY is exported ONLY for
#     judge-scored tasks (arenahardwriting, healthbench) and per the benchmark
#     rules may be used by evaluate.py exclusively — kapso's own LLM roles are
#     pinned to Anthropic in benchmarks/posttrain/config.yaml.
#   - the harness enforces the deadline with `timeout` (TERM, then KILL +30s)
#
# Kapso lives in its own venv (see containers/kapso.def) so its dependencies
# never touch the pinned training environment the agent works in.

unset GEMINI_API_KEY
unset CODEX_API_KEY

# Claude Max subscription auth: run_task.sh copies agents/kapso/oauth_token to
# the job home when present (same mechanism as the upstream claude_non_api
# agent). Prefer it over the API key to avoid auth conflicts.
if [ -f /home/ben/oauth_token ]; then
    CLAUDE_CODE_OAUTH_TOKEN="$(cat /home/ben/oauth_token)"
    export CLAUDE_CODE_OAUTH_TOKEN
    unset ANTHROPIC_API_KEY
fi

# Claude Code bash-tool ceiling: training commands run for hours.
export BASH_MAX_TIMEOUT_MS="36000000"
export BASH_DEFAULT_TIMEOUT_MS="36000000"

# The container runs as root and Claude Code refuses
# --dangerously-skip-permissions under root unless it knows it's sandboxed.
export IS_SANDBOX=1

exec /opt/kapso/venv/bin/expert-posttrain \
    --task-dir "$PWD" \
    --prompt-env PROMPT \
    --coding-model "${AGENT_CONFIG:-claude-opus-4-6}" \
    --mode POSTTRAIN
