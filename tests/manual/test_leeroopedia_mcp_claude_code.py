# Manual production test: Claude Code -> Leeroopedia's hosted MCP server,
# through Kapso's own gate config. Exercises all 8 tools.
#
# The MCP config is built exactly as a campaign session receives it
# (get_mcp_config(["leeroopedia"]) -> {"type": "http", "url": ...}), then
# `claude -p` runs once per tool with that config, --strict-mcp-config and
# only that tool allowed. A tool passes when the init event reports the
# server connected, the tool was actually called, and its result is neither
# an authentication error nor empty.
#
# Requires:
#   - `claude` on PATH with a working login (CLAUDE_CODE_OAUTH_TOKEN in the
#     environment is forwarded as-is)
#   - LEEROOPEDIA_API_KEY in the repo .env or the environment
#
# Run all 8 tools:        python tests/manual/test_leeroopedia_mcp_claude_code.py
# Run specific tools:     python tests/manual/test_leeroopedia_mcp_claude_code.py get_page search_knowledge

import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from kapso.gated_mcp import GATES, get_mcp_config

PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Agentic tools run 30-180s on the API side, plus CLI startup and reasoning.
TOOL_TIMEOUT = 600
MIN_RESULT_LENGTH = 50
LOG_PATH = Path(tempfile.gettempdir()) / "leeroopedia_mcp_claude_code_results.log"

MCP_SERVER_NAME = GATES["leeroopedia"].server_name
MCP_TOOL_PREFIX = f"mcp__{MCP_SERVER_NAME}__"
TOOL_NAMES = list(GATES["leeroopedia"].tools)

# (tool_name, prompt, description) — each prompt names the one tool to call.
TEST_CASES = [
    (
        "get_page",
        (
            f"Use the {MCP_TOOL_PREFIX}get_page tool to retrieve the page with "
            f'page_id "Heuristic/Huggingface_Alignment_handbook_QLoRA_Learning_Rate_Scaling". '
            "Return the full page content you receive from the tool."
        ),
        "Direct page retrieval by exact ID (no agent)",
    ),
    (
        "search_knowledge",
        (
            f"Use the {MCP_TOOL_PREFIX}search_knowledge tool with "
            f'query="What is LoRA and how does it work?" and '
            f'context="Focus on parameter efficiency and rank selection". '
            "Return the full response from the tool."
        ),
        "Research librarian synthesis with citations",
    ),
    (
        "build_plan",
        (
            f"Use the {MCP_TOOL_PREFIX}build_plan tool with "
            f'goal="Fine-tune Llama-3 8B on a custom instruction dataset" and '
            f'constraints="Single A100 80GB GPU, complete within 4 hours". '
            "Return the full response from the tool."
        ),
        "Step-by-step ML execution plan",
    ),
    (
        "review_plan",
        (
            f"Use the {MCP_TOOL_PREFIX}review_plan tool with "
            f'proposal="1. Load Llama-3 8B in 4-bit with QLoRA\n'
            f"2. Use LoRA rank 64, alpha 128\n"
            f"3. Train for 3 epochs with lr=2e-4\n"
            f"4. Use batch size 4 with gradient accumulation 8\n"
            f'5. Evaluate on held-out set" and '
            f'goal="Fine-tune Llama-3 8B for instruction following". '
            "Return the full response from the tool."
        ),
        "Plan review with approvals, risks, suggestions",
    ),
    (
        "verify_code_math",
        (
            f"Use the {MCP_TOOL_PREFIX}verify_code_math tool with "
            f'concept_name="LoRA low-rank adaptation" and '
            "code_snippet containing this Python code:\n\n"
            "```python\n"
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class LoRALayer(nn.Module):\n"
            "    def __init__(self, in_dim, out_dim, rank=4, alpha=1):\n"
            "        super().__init__()\n"
            "        self.A = nn.Parameter(torch.randn(in_dim, rank))\n"
            "        self.B = nn.Parameter(torch.zeros(rank, out_dim))\n"
            "        self.scale = alpha / rank\n\n"
            "    def forward(self, x):\n"
            "        return x @ self.A @ self.B * self.scale\n"
            "```\n\n"
            "Return the full response from the tool."
        ),
        "Code/math verification with Pass/Fail verdict",
    ),
    (
        "diagnose_failure",
        (
            f"Use the {MCP_TOOL_PREFIX}diagnose_failure tool with "
            f'symptoms="Training loss goes to NaN after ~100 steps during QLoRA fine-tuning of Llama-3 8B" and '
            f'logs="Step 98: loss=0.853\nStep 99: loss=1.247\nStep 100: loss=nan\n'
            f'Step 101: loss=nan\nRuntimeWarning: overflow encountered in float16". '
            "Return the full response from the tool."
        ),
        "Failure diagnosis with fix and prevention",
    ),
    (
        "propose_hypothesis",
        (
            f"Use the {MCP_TOOL_PREFIX}propose_hypothesis tool with "
            f'current_status="Fine-tuned Llama-3 8B with QLoRA on instruction data. '
            f"Training loss converged to 0.8 but eval performance is poor — "
            f'model repeats itself and ignores instructions." and '
            f'recent_experiments="Tried rank 16 and rank 64, both show same repetition. '
            f'Increased dataset to 50k samples, no improvement." '
            "Return the full response from the tool."
        ),
        "Ranked hypotheses with KB-grounded rationale",
    ),
    (
        "query_hyperparameter_priors",
        (
            f"Use the {MCP_TOOL_PREFIX}query_hyperparameter_priors tool with "
            f'query="Recommended learning rate, rank, and alpha for LoRA fine-tuning Llama-3 8B". '
            "Return the full response from the tool."
        ),
        "Hyperparameter suggestion table with justification",
    ),
]


def masked(text: str) -> str:
    """The key must never land in a log: keep only the kpsk_ prefix."""
    key = os.environ["LEEROOPEDIA_API_KEY"]
    return text.replace(key, key[:13] + "…")


def create_mcp_config() -> Path:
    """The MCP config a campaign session gets for the leeroopedia gate,
    written outside the repo (it carries the key)."""
    servers, _ = get_mcp_config(
        ["leeroopedia"], gate_failure_policy="error", include_base_tools=False,
    )
    path = Path(tempfile.mkdtemp(prefix="leeroopedia_mcp_")) / "mcp_config.json"
    path.write_text(json.dumps({"mcpServers": servers}, indent=2))
    return path


def run_claude_with_tool(
    prompt: str, config_path: Path, tool_name: str, log_write
) -> dict:
    """One `claude -p` session allowed exactly this tool; returns what the
    stream showed: server connected, tool called, its raw result, cost."""
    allowed = f"{MCP_TOOL_PREFIX}{tool_name}"
    cmd = [
        "claude", "-p", prompt,
        "--mcp-config", str(config_path),
        "--strict-mcp-config",
        "--allowedTools", allowed,
        "--output-format", "stream-json",
        "--verbose",
    ]
    start = time.time()
    proc = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True,
        timeout=TOOL_TIMEOUT, env={**os.environ},
    )
    elapsed = time.time() - start

    events = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        log_write(f"  STREAM: {masked(line)}")
        events.append(json.loads(line))

    connected = any(
        e.get("type") == "system" and e.get("subtype") == "init"
        and any(
            s.get("name") == MCP_SERVER_NAME and s.get("status") == "connected"
            for s in e.get("mcp_servers", [])
        )
        for e in events
    )
    called = any(
        block.get("type") == "tool_use" and block.get("name") == allowed
        for e in events if e.get("type") == "assistant"
        for block in e["message"]["content"]
    )
    tool_result = ""
    for e in events:
        if e.get("type") != "user":
            continue
        for block in e["message"]["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                content = block.get("content", "")
                if isinstance(content, list):
                    content = "".join(c.get("text", "") for c in content)
                tool_result = content
    final = [e for e in events if e.get("type") == "result"]
    cost = final[-1].get("total_cost_usd", 0.0) if final else 0.0
    auth_error = "Authentication error" in tool_result
    return {
        "returncode": proc.returncode,
        "stderr": proc.stderr,
        "elapsed": elapsed,
        "cost": cost,
        "connected": connected,
        "called": called,
        "tool_result": tool_result,
        "auth_error": auth_error,
        "events": events,
        "success": (
            proc.returncode == 0 and connected and called and not auth_error
            and len(tool_result) >= MIN_RESULT_LENGTH
        ),
    }


def run_tests(tool_filter=None) -> bool:
    if not os.environ.get("LEEROOPEDIA_API_KEY"):
        print("ERROR: LEEROOPEDIA_API_KEY not set (repo .env or environment).")
        return False
    config_path = create_mcp_config()
    cases = [c for c in TEST_CASES if not tool_filter or c[0] in tool_filter]

    log = open(LOG_PATH, "w", encoding="utf-8")
    def log_write(text: str) -> None:
        log.write(text + "\n")
        log.flush()

    log_write("Leeroopedia hosted MCP — Claude Code production test")
    log_write(f"Generated: {datetime.now().isoformat()}")
    log_write(f"MCP config: {masked(config_path.read_text())}")
    print(f"MCP config (as a campaign session gets it):\n{masked(config_path.read_text())}")
    print(f"Log: {LOG_PATH}\n")

    results = []
    total_cost = 0.0
    for index, (tool_name, prompt, description) in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {tool_name} — {description}")
        log_write("\n" + "=" * 80 + f"\n[{index}/{len(cases)}] {tool_name}\nPROMPT:\n{prompt}\n" + "-" * 80)
        result = run_claude_with_tool(prompt, config_path, tool_name, log_write)
        total_cost += result["cost"]
        verdict = "PASS" if result["success"] else "FAIL"
        head = result["tool_result"][:160].replace("\n", " ")
        print(
            f"  {verdict}  connected={result['connected']} called={result['called']} "
            f"auth_error={result['auth_error']} result_chars={len(result['tool_result'])} "
            f"{result['elapsed']:.0f}s ${result['cost']:.3f}"
        )
        print(f"  result head: {head}")
        if not result["success"]:
            print(f"  stderr: {result['stderr'][-400:]}")
        log_write(f"VERDICT: {verdict}\nTOOL RESULT:\n{result['tool_result']}\n")
        results.append((tool_name, result["success"]))

    passed = sum(1 for _, ok in results if ok)
    summary = f"\n{passed}/{len(results)} tools passed; total cost ${total_cost:.3f}"
    print(summary)
    log_write(summary)
    log.close()
    return passed == len(results)


if __name__ == "__main__":
    ok = run_tests(set(sys.argv[1:]) or None)
    sys.exit(0 if ok else 1)
