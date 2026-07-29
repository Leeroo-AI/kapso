#!/usr/bin/env bash
# =============================================================================
# setup_box.sh — one-time box setup = the GOLDEN IMAGE contents.
#
# Run ONCE on a fresh GCP Deep Learning VM, then snapshot the disk to make a
# reusable image so future competition boxes boot ready — no 20-40 min install
# against the competition clock. Everything here is deterministic and
# TASK-AGNOSTIC. NO secrets are baked; bootstrap.sh injects them per boot.
#
# Usage (on the fresh VM):
#   GITHUB_PAT=<pat> bash setup_box.sh
# =============================================================================
set -euo pipefail

: "${GITHUB_PAT:?need GITHUB_PAT to clone the private kapso repo (injected for the clone, never persisted)}"
KAPSO_BRANCH="${KAPSO_BRANCH:-worktree-ioai-2025}"
VENV="${VENV:-$HOME/kapso-venv}"
CONDA_PY="${CONDA_PY:-/opt/conda/bin/python}"   # DLVM base python (>=3.11, kaggle CLI 2.x needs it)

echo "### 1/5 node + codex CLI (the DLVM ships no node; ensemble + coding agent need it)"
if ! command -v node >/dev/null 2>&1; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
  sudo apt-get install -y nodejs
fi
sudo npm install -g @openai/codex
codex --version

echo "### 2/5 isolated python venv (py>=3.11)"
"$CONDA_PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip wheel

echo "### 3/5 clone kapso (@$KAPSO_BRANCH) — PAT used for the clone, then dropped from the remote"
if [ ! -d "$HOME/kapso/.git" ]; then
  git clone "https://x-access-token:${GITHUB_PAT}@github.com/leeroo/kapso.git" "$HOME/kapso"
fi
cd "$HOME/kapso"
git remote set-url origin https://github.com/leeroo/kapso.git
git -c "http.extraheader=AUTHORIZATION: basic $(printf 'x-access-token:%s' "$GITHUB_PAT" | base64 -w0)" \
    fetch origin "$KAPSO_BRANCH"
git checkout "$KAPSO_BRANCH"

echo "### 4/5 deps: kapso requirements + torch triple (clean, dodges the torchvision::nms ABI trap) + audio + kaggle"
pip install -r requirements.txt
pip install --ignore-installed torch torchvision torchaudio
pip install transformers soundfile librosa kaggle
pip install -e . --no-deps

echo "### 5/5 kaggle on PATH"
mkdir -p "$HOME/.local/bin"
ln -sf "$VENV/bin/kaggle" "$HOME/.local/bin/kaggle"

echo
echo "DONE. Snapshot this disk now → golden image. Secrets are NOT baked;"
echo "bootstrap.sh injects them (and pulls fresh code) at boot."
