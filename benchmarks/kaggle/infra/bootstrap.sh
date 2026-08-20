#!/usr/bin/env bash
# =============================================================================
# bootstrap.sh — per-boot: pull fresh code + inject secrets + smoke-test.
#
# Run on a box provisioned from the golden image, BEFORE the competition clock.
# Secrets arrive as env vars (GCP instance metadata or a secure copy) and are
# written to disk here — NEVER baked into the image. Ships ONLY the safe vars:
# the Bedrock trio (AWS_BEARER_TOKEN_BEDROCK / CLAUDE_CODE_USE_BEDROCK /
# ANTHROPIC_MODEL) must never reach the box — they hijack Claude Code onto
# Bedrock and silently change which model runs.
#
# Required env: OPENAI_API_KEY, CLAUDE_CODE_OAUTH_TOKEN, GITHUB_PAT,
#               CODEX_AUTH_JSON (file contents), KAGGLE_ACCESS_TOKEN (contents)
# Optional:     FIREWORKS_API_KEY (the oss_claude_code ideation member fails
#               without it), HF_TOKEN, KAPSO_COMMIT (default worktree-ioai-2025)
# =============================================================================
set -euo pipefail
: "${OPENAI_API_KEY:?}"; : "${CLAUDE_CODE_OAUTH_TOKEN:?}"; : "${GITHUB_PAT:?}"
: "${CODEX_AUTH_JSON:?}"; : "${KAGGLE_ACCESS_TOKEN:?}"
KAPSO_COMMIT="${KAPSO_COMMIT:-worktree-ioai-2025}"
# PAT in the URL for the authenticated fetch, stripped again right after: the
# `http.extraheader` basic-auth form does not authenticate against this repo.
PAT_URL="https://x-access-token:${GITHUB_PAT}@github.com/Leeroo-AI/kapso.git"
CLEAN_URL="https://github.com/Leeroo-AI/kapso.git"
# System python3 owns torch on this image (see setup_box.sh); no venv exists.
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/kapso/src:$HOME/kapso"

echo "### code -> $KAPSO_COMMIT (bake the env, pull the code fresh)"
cd "$HOME/kapso"
git remote set-url origin "$PAT_URL"
git fetch origin --tags --quiet
git checkout "$KAPSO_COMMIT" --quiet
git pull --ff-only --quiet origin "$KAPSO_COMMIT" || true
git remote set-url origin "$CLEAN_URL"
echo "  on $(git branch --show-current 2>/dev/null || echo detached) @ $(git rev-parse --short HEAD)"

echo "### curated .env (safe vars ONLY — never the Bedrock trio)"
umask 077
cat > "$HOME/kapso/.env" <<EOF
OPENAI_API_KEY=${OPENAI_API_KEY}
CLAUDE_CODE_OAUTH_TOKEN=${CLAUDE_CODE_OAUTH_TOKEN}
FIREWORKS_API_KEY=${FIREWORKS_API_KEY:-}
HF_TOKEN=${HF_TOKEN:-}
EOF

echo "### auth files (codex token rotates; kaggle token)"
mkdir -p "$HOME/.codex" "$HOME/.kaggle"
printf '%s' "$CODEX_AUTH_JSON"     > "$HOME/.codex/auth.json"
printf '%s' "$KAGGLE_ACCESS_TOKEN" > "$HOME/.kaggle/access_token"

echo "### smoke-test (fail the BOOT here, not 40 min into the run)"
python3 - <<'PY'
import torch, kapso
assert torch.cuda.is_available(), "CUDA not available"
print("  torch", torch.__version__, "| GPUs", torch.cuda.device_count())
print("  kapso", kapso.__file__)
PY
kaggle competitions list >/dev/null 2>&1 && echo "  kaggle auth ok"
codex --version >/dev/null 2>&1 && echo "  codex ok"
claude --version >/dev/null 2>&1 && echo "  claude ok"
echo "BOOTSTRAP OK: $(git rev-parse --short HEAD), $(python3 -c 'import torch;print(torch.cuda.device_count())') GPUs ready."
