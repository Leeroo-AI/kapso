#!/usr/bin/env bash
# =============================================================================
# setup_box.sh — one-time box setup = the GOLDEN IMAGE contents.
#
# Run ONCE on a fresh GCP Deep Learning VM (image family
# pytorch-2-9-cu129-ubuntu-2204-nvidia-580), then snapshot the disk to make a
# reusable image so future competition boxes boot ready — no 20-40 min install
# against the competition clock. Everything here is deterministic and
# TASK-AGNOSTIC. NO secrets are baked; bootstrap.sh injects them per boot.
#
# Image facts this script is built around (verified live 2026-07-29):
#   - There is NO /opt/conda. The system /usr/bin/python3 (3.10) ALREADY ships
#     torch 2.9.1+cu129 seeing every GPU -> inherit it, install with --user.
#     Never rebuild the torch triple; that is the slow, ABI-fragile path.
#   - `python3 -m venv` fails (no ensurepip) and is unnecessary for the same
#     reason.
#   - `pip install -e .` fails (old pip lacks the PEP 660 build_editable hook)
#     -> kapso is used via PYTHONPATH instead (see run_competition.sh).
#   - kaggle CLI 2.x needs py>=3.11; py3.10 resolves the pre-KGAT 1.7 which
#     crashes at import -> install via `uv tool install --python 3.11`.
#   - Both CLIs are needed: codex (implementation/selector/feedback) AND claude
#     (ideation ensemble member + lens planner).
#
# Usage (on the fresh VM):
#   GITHUB_PAT=<pat> bash setup_box.sh
# =============================================================================
set -euo pipefail

: "${GITHUB_PAT:?need GITHUB_PAT to clone the private kapso repo (used for the clone, never persisted)}"
KAPSO_BRANCH="${KAPSO_BRANCH:-worktree-ioai-2025}"
export PATH="$HOME/.local/bin:$PATH"

echo "### 1/6 node + the two agent CLIs (the DLVM ships no node)"
if ! command -v node >/dev/null 2>&1; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
  sudo apt-get install -y nodejs
fi
sudo npm install -g @openai/codex @anthropic-ai/claude-code
echo "  codex:  $(codex --version)"
echo "  claude: $(claude --version)"

echo "### 2/6 verify the system python already has CUDA torch (we inherit it)"
python3 - <<'PY'
import sys, torch
assert sys.version_info >= (3, 10), sys.version
assert torch.cuda.is_available(), "system torch cannot see a GPU"
print(f"  python {sys.version.split()[0]} | torch {torch.__version__} | GPUs {torch.cuda.device_count()}")
PY

echo "### 3/6 kaggle CLI via uv on py3.11 (py3.10 would pull the broken pre-KGAT 1.7)"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install kaggle==2.2.3 --python 3.11 --force  # pinned: match the dev box; 2.2.4 relocates the entry point
echo "  kaggle: $(kaggle --version)"

echo "### 4/6 clone kapso @$KAPSO_BRANCH — PAT used for the clone, then dropped from the remote"
# Clone the BRANCH directly: a plain clone lands on main (no benchmarks/kaggle
# work) and the follow-up authenticated fetch is where things break. The
# `http.extraheader` basic-auth form does NOT authenticate against this repo
# (git falls through to an interactive prompt -> "could not read Username"), so
# the PAT goes in the URL for the one authenticated operation, then is stripped.
if [ ! -d "$HOME/kapso/.git" ]; then
  git clone --branch "$KAPSO_BRANCH" \
    "https://x-access-token:${GITHUB_PAT}@github.com/Leeroo-AI/kapso.git" "$HOME/kapso"
fi
cd "$HOME/kapso"
git remote set-url origin https://github.com/Leeroo-AI/kapso.git
test "$(git branch --show-current)" = "$KAPSO_BRANCH" \
  || { echo "on $(git branch --show-current), expected $KAPSO_BRANCH"; exit 1; }
echo "  branch $(git branch --show-current) @ $(git rev-parse --short HEAD)"

echo "### 5/6 kapso deps (--user; torch inherited, NOT reinstalled)"
# torchaudio must match the inherited torch exactly (the image ships
# torchaudio 2.11 against torch 2.9.1 — the ABI mismatch makes
# `import transformers` crash with "undefined symbol: torch_library_impl"
# via transformers.loss.loss_rnnt).
TORCH_V=$(python3 -c "import torch; print(torch.__version__.split('+')[0])")
TA_IDX=$(python3 -c "import torch; c=torch.version.cuda; print('https://download.pytorch.org/whl/cu'+c.replace('.','') if c else 'https://download.pytorch.org/whl/cpu')")
python3 -m pip install --user -q "torchaudio==${TORCH_V}" --index-url "$TA_IDX"
python3 -m pip install --user -q \
  litellm==1.75.0 openai PyYAML python-dotenv GitPython scipy "mcp>=1.9,<2" \
  transformers soundfile librosa
python3 -c "import transformers; print('  transformers import OK (torchaudio ABI matched)')"

echo "### 6/6 verify kapso imports via PYTHONPATH (no editable install on this image)"
PYTHONPATH="$HOME/kapso/src:$HOME/kapso" python3 - <<'PY'
import kapso, benchmarks.kaggle.preflight, benchmarks.kaggle.runner  # noqa: F401
from kapso.execution.coding_agents.factory import CodingAgentFactory as F
assert F.is_available("codex") and F.is_available("claude_code")
print("  kapso + benchmarks import OK; codex + claude_code registered")
PY

echo
echo "DONE. Snapshot this disk now → golden image. Secrets are NOT baked;"
echo "bootstrap.sh injects them (and pulls fresh code) at boot."
