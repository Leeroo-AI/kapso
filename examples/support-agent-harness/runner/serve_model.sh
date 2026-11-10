#!/usr/bin/env bash
# Serve the agent's model and the embedding model with vLLM, on a machine with one 80 GB GPU:
#
#   chat         Qwen/Qwen3.8-27B-FP8 as `qwen`, with tool calls and thinking          <chat port>, default 18000
#   embeddings   Qwen/Qwen3-Embedding-8B as `qwen-embed`, for the runner's dense search  <embedding port>, default 18001
#
# The chat server's flags are the fixed conditions of repo/bench/fixed_conditions.json: the model's native
# context of 262,144 tokens, an FP8 key-value cache (about 818k tokens beside the embeddings, room for a few
# full-length conversations next to the normal ones) and the Qwen tool-call and reasoning parsers.
# vLLM downloads both models from Hugging Face on the first start; everything lands under ~/model-serving.
#
# Needs uv, a C toolchain, the Python headers and ninja on the machine (Debian and Ubuntu: apt install
# build-essential python3-dev ninja-build): a GPU kernel helper is compiled at start-up.
#
# The servers listen on <bind address> without authentication. The default, every address, is what lets
# another machine or a container reach them: keep the ports closed to the outside, or pass 127.0.0.1 when
# nothing but this machine needs them.
#
# Usage: bash runner/serve_model.sh [<chat port> [<embedding port> [<bind address>]]]
set -euo pipefail
CHAT_PORT="${1:-18000}"
EMBED_PORT="${2:-18001}"
BIND="${3:-0.0.0.0}"
ROOT="$HOME/model-serving"
command -v uv > /dev/null || { echo "uv is needed: https://docs.astral.sh/uv/"; exit 1; }
mkdir -p "$ROOT/hf"
export HF_HOME="$ROOT/hf"

if [ ! -x "$ROOT/vllm/bin/vllm" ]; then
  uv venv --python 3.12 "$ROOT/vllm"
  uv pip install --python "$ROOT/vllm/bin/python" vllm==0.32.0 openai ninja cmake
fi
export PATH="$ROOT/vllm/bin:$PATH"   # vLLM looks ninja up on PATH

# The FP8 block-scale kernel of one backend is compiled with nvcc at start-up and failed on the recorded
# machines' image ("cubin.empty"); these two switches select the path that needs no nvcc.
export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0 VLLM_USE_DEEP_GEMM=0

wait_for() {   # <port> <served name> <log file> <tries of five seconds>
  for _ in $(seq 1 "$4"); do
    if curl --silent --max-time 5 "localhost:$1/v1/models" | grep -q "\"$2\""; then return 0; fi
    if grep -q "Engine core initialization failed" "$3"; then
      echo "the server on port $1 failed to start:"; grep -E "RuntimeError|ValueError|Assertion|Error" "$3" | tail -3 | cut -c1-240; return 1
    fi
    sleep 5
  done
  echo "the server on port $1 did not list $2 in time; see $3"; return 1
}

# --max-num-seqs: the model's linear-attention layers take one cache block per sequence, and the default
# of 1,024 sequences is more than fit.
nohup vllm serve Qwen/Qwen3.8-27B-FP8 --served-model-name qwen --host "$BIND" --port "$CHAT_PORT" \
  --max-model-len 262144 --kv-cache-dtype fp8 --gpu-memory-utilization 0.72 --max-num-seqs 64 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  > "$ROOT/vllm_chat.log" 2>&1 < /dev/null &
wait_for "$CHAT_PORT" qwen "$ROOT/vllm_chat.log" 360

nohup vllm serve Qwen/Qwen3-Embedding-8B --served-model-name qwen-embed text-embedding-3-large --host "$BIND" --port "$EMBED_PORT" \
  --runner pooling --gpu-memory-utilization 0.25 --max-model-len 8192 \
  > "$ROOT/vllm_embed.log" 2>&1 < /dev/null &
wait_for "$EMBED_PORT" qwen-embed "$ROOT/vllm_embed.log" 240

echo "chat:       http://localhost:$CHAT_PORT/v1 serves qwen ($(grep -E 'KV cache size' "$ROOT/vllm_chat.log" | tail -1 | cut -c1-120))"
echo "embeddings: http://localhost:$EMBED_PORT/v1 serves qwen-embed"
echo "logs: $ROOT/vllm_chat.log, $ROOT/vllm_embed.log"
