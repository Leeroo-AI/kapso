#!/usr/bin/env bash
# Stand the conversation runner up in a directory of its own, outside the repository:
#
#   <runner dir>/tau2-bench-2.0.0/      τ²-bench 2.0.0 (Sierra Research, MIT) with its knowledge extra
#   <runner dir>/run_conversations.py   the script bench/run_dev.py calls
#
# and, under ~/.local, the sandbox runtime the runner's shell tool runs in. Needs uv, npm, curl, tar and, for
# the sandbox on Linux, bubblewrap, socat and ripgrep (Debian and Ubuntu: apt install bubblewrap socat ripgrep).
#
# One line of the runner's source is changed: the model that judges a natural-language assertion is routed
# through OpenRouter, because the runner's embedder takes over the OpenAI SDK's variables (they point at the
# local embedding server), so the stock OpenAI route cannot be reached. One of the 20 development
# conversations carries such an assertion. Without OPENROUTER_API_KEY in the customer env file that
# conversation ends in an error and counts as failed, which the shipped brief's last rule assumes.
# The line is written exactly as it stood in our runner, so the source hashes as runner/recorded_hashes.json has it.
#
# Usage: bash runner/install.sh <runner dir>
set -euo pipefail
RUNNER_DIR="${1:?usage: install.sh <runner dir>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
for tool in uv npm curl tar bwrap socat rg python3; do
  command -v "$tool" > /dev/null || { echo "missing on PATH: $tool"; exit 1; }
done
mkdir -p "$RUNNER_DIR"
RUNNER_DIR="$(cd "$RUNNER_DIR" && pwd)"
SOURCE="$RUNNER_DIR/tau2-bench-2.0.0"

if [ ! -d "$SOURCE" ]; then
  curl --location --fail --silent --show-error --output "$RUNNER_DIR/tau2-bench.tar.gz" \
    https://codeload.github.com/sierra-research/tau2-bench/tar.gz/refs/tags/v2.0.0
  tar -C "$RUNNER_DIR" -xzf "$RUNNER_DIR/tau2-bench.tar.gz"
  rm "$RUNNER_DIR/tau2-bench.tar.gz"
fi

# The runner's own environment, from its lock file as shipped (the lock file is one of the hashed conditions).
(cd "$SOURCE" && uv sync --frozen --extra knowledge)

# The sandbox the runner's shell tool runs in (`srt` under ~/.local/bin).
npm install --global --silent --prefix "$HOME/.local" @anthropic-ai/sandbox-runtime@0.0.23

python3 - "$SOURCE/src/tau2/config.py" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
stock = 'DEFAULT_LLM_NL_ASSERTIONS = "gpt-5.4-mini-2026-03-17"\n'
routed = ('DEFAULT_LLM_NL_ASSERTIONS = "openrouter/openai/gpt-5.4-mini"'
          '  # demo: judge routed through OpenRouter (was gpt-5.4-mini-2026-03-17)\n')
text = path.read_text(encoding="utf-8")
if routed in text:
    print("the assertion judge is already routed through OpenRouter")
elif text.count(stock) == 1:
    path.write_text(text.replace(stock, routed), encoding="utf-8")
    print("routed the assertion judge through OpenRouter (one line of src/tau2/config.py)")
else:
    raise SystemExit(f"{path}: the stock judge line was not found exactly once; this is not τ²-bench 2.0.0 as released")
PY

cp "$HERE/run_conversations.py" "$RUNNER_DIR/run_conversations.py"

definitions="$(python3 -c 'import json, sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' \
  "$SOURCE/data/tau2/domains/banking_knowledge/tasks.json")"
echo "runner installed at $RUNNER_DIR"
echo "  source: $SOURCE ($(cd "$SOURCE" && uv run python -c 'import tau2; print("tau2 imports")' 2> /dev/null))"
echo "  the banking domain's registry holds $definitions task definitions"
echo "next: python runner/split.py and runner/seal.py to keep a held-out set, then python runner/configure.py $RUNNER_DIR"
