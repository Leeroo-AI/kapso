# Test: Leeroopedia MCP — Claude Code Production Integration (All 8 Tools)
#
# Complete end-to-end production test that connects Claude Code CLI to the
# leeroopedia-mcp server (following the README setup) and exercises all 8 tools.
#
# This verifies the full production path:
#   Claude Code CLI → MCP stdio → leeroopedia-mcp server → api.leeroopedia.com
#   → Celery → backend_search → AgenticKGSearch → KG + Weaviate
#
# Streams full Claude Code logs live: thinking, tool calls, tool results, text.
#
# Requires:
#   - Claude Code CLI installed (`claude` on PATH)
#   - AWS_BEARER_TOKEN_BEDROCK env var (for Claude Code via Bedrock)
#   - LEEROOPEDIA_API_KEY env var (in .env or environment)
#   - leeroopedia-mcp package installed (`pip install -e leeroopedia-mcp`)
#   - api.leeroopedia.com reachable
#
# Run all 8 tools:
#   python tests/test_leeroopedia_mcp_claude_code.py
#
# Run only get_page (fast, no agent):
#   python tests/test_leeroopedia_mcp_claude_code.py get_page
#
# Run specific tools:
#   python tests/test_leeroopedia_mcp_claude_code.py get_page search_knowledge

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import threading
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --- ANSI colors ---
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_GREEN = "\033[32m"
C_RED = "\033[31m"
C_YELLOW = "\033[33m"
C_BLUE = "\033[34m"
C_CYAN = "\033[36m"
C_MAGENTA = "\033[35m"

# --- Configuration ---
API_KEY = os.getenv("LEEROOPEDIA_API_KEY", "")
BEDROCK_TOKEN = os.getenv("AWS_BEARER_TOKEN_BEDROCK", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Timeout per tool invocation (agentic tools can take 30-180s on the API side,
# plus Claude Code startup + MCP handshake + Claude reasoning overhead)
TOOL_TIMEOUT = 600  # 10 minutes

# Minimum response length
MIN_RESPONSE_LENGTH = 50

# Log file
LOG_PATH = Path(__file__).parent / "leeroopedia_mcp_claude_code_results.log"

# MCP server name (must match the key in mcpServers config)
MCP_SERVER_NAME = "leeroopedia"

# Claude Code allowedTools — all 8 leeroopedia MCP tools
MCP_TOOL_PREFIX = f"mcp__{MCP_SERVER_NAME}__"
TOOL_NAMES = [
    "search_knowledge",
    "build_plan",
    "review_plan",
    "verify_code_math",
    "diagnose_failure",
    "propose_hypothesis",
    "query_hyperparameter_priors",
    "get_page",
]
ALLOWED_TOOLS = ",".join(f"{MCP_TOOL_PREFIX}{t}" for t in TOOL_NAMES)

# Test cases: (tool_name, prompt_for_claude, description)
# Each prompt explicitly instructs Claude to call the specific MCP tool.
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


def create_mcp_config(api_key: str) -> Path:
    """
    Create a temporary MCP config file following the leeroopedia-mcp README.

    Returns:
        Path to the temporary config file.
    """
    config = {
        "mcpServers": {
            MCP_SERVER_NAME: {
                "command": "leeroopedia-mcp",
                "env": {
                    "LEEROOPEDIA_API_KEY": api_key,
                },
            }
        }
    }

    config_path = Path(tempfile.gettempdir()) / "test_leeroopedia_mcp_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    return config_path


def format_stream_event(event: dict) -> str:
    """
    Format a stream-json event for live console output.

    Returns formatted string to print, or empty string to skip.
    """
    etype = event.get("type", "")

    # --- System / init messages ---
    if etype == "system":
        msg = event.get("message", "")
        subtype = event.get("subtype", "")
        if subtype == "init":
            tools = event.get("tools", [])
            mcp_tools = [t for t in tools if t.startswith("mcp__")]
            return (
                f"  {C_DIM}[system] session initialized | "
                f"MCP tools: {len(mcp_tools)}{C_RESET}"
            )
        return f"  {C_DIM}[system] {msg or subtype}{C_RESET}"

    # --- Assistant message (thinking, text, tool_use) ---
    if etype == "assistant":
        message = event.get("message", {})
        content_blocks = message.get("content", [])
        lines = []
        for block in content_blocks:
            btype = block.get("type", "")

            if btype == "thinking":
                thinking = block.get("thinking", "")
                # Show first 300 chars of thinking
                preview = thinking[:300].replace("\n", "\n  │ ")
                lines.append(f"  {C_MAGENTA}💭 [thinking]{C_RESET}")
                lines.append(f"  {C_DIM}│ {preview}")
                if len(thinking) > 300:
                    lines.append(f"  │ ... ({len(thinking)} chars total)")
                lines.append(f"  {C_RESET}")

            elif btype == "tool_use":
                tool_name = block.get("name", "?")
                tool_input = block.get("input", {})
                tool_id = block.get("id", "")
                input_str = json.dumps(tool_input, indent=2)
                # Truncate large inputs
                if len(input_str) > 500:
                    input_str = input_str[:500] + f"\n  ... ({len(input_str)} chars)"
                lines.append(
                    f"  {C_YELLOW}🔧 [tool_call] {C_BOLD}{tool_name}{C_RESET}"
                    f"{C_DIM} ({tool_id}){C_RESET}"
                )
                for input_line in input_str.split("\n"):
                    lines.append(f"  {C_DIM}  {input_line}{C_RESET}")

            elif btype == "text":
                text = block.get("text", "")
                preview = text[:500].replace("\n", "\n  │ ")
                lines.append(f"  {C_CYAN}📝 [text]{C_RESET}")
                lines.append(f"  │ {preview}")
                if len(text) > 500:
                    lines.append(f"  │ ... ({len(text)} chars total)")

        return "\n".join(lines) if lines else ""

    # --- Tool result ---
    if etype == "tool":
        content_blocks = event.get("content", [])
        tool_id = event.get("tool_use_id", "")
        lines = []
        for block in content_blocks:
            btype = block.get("type", "")
            if btype == "text":
                text = block.get("text", "")
                preview = text[:800].replace("\n", "\n  │ ")
                lines.append(
                    f"  {C_BLUE}📦 [tool_result]{C_RESET}"
                    f"{C_DIM} ({tool_id}){C_RESET}"
                )
                lines.append(f"  │ {preview}")
                if len(text) > 800:
                    lines.append(f"  │ ... ({len(text)} chars total)")
        return "\n".join(lines) if lines else ""

    # --- Result (final) ---
    if etype == "result":
        cost = event.get("cost_usd", 0)
        duration = event.get("duration_ms", 0)
        turns = event.get("num_turns", 0)
        is_error = event.get("is_error", False)
        status = f"{C_RED}ERROR{C_RESET}" if is_error else f"{C_GREEN}OK{C_RESET}"
        return (
            f"  {C_BOLD}[result]{C_RESET} status={status} "
            f"turns={turns} cost=${cost:.4f} "
            f"duration={duration/1000:.1f}s"
        )

    # --- Catch-all for unknown event types ---
    if etype:
        return f"  {C_DIM}[{etype}]{C_RESET}"
    return ""


def run_claude_with_tool(
    prompt: str, config_path: Path, tool_name: str, log_write
) -> dict:
    """
    Run Claude Code CLI with MCP config, streaming events live to console.

    Uses --output-format stream-json for full visibility into thinking,
    tool calls, and tool results.

    Args:
        prompt: The prompt instructing Claude to use the specific tool.
        config_path: Path to the MCP config JSON.
        tool_name: Tool name (for logging).
        log_write: Function to write to the log file.

    Returns:
        dict with keys: success, output, stderr, returncode, elapsed, events
    """
    allowed = f"{MCP_TOOL_PREFIX}{tool_name}"

    cmd = [
        "claude",
        "-p", prompt,
        "--mcp-config", str(config_path),
        "--output-format", "stream-json",
        "--verbose",
        "--allowedTools", allowed,
    ]

    # Build env with Bedrock credentials for Claude Code
    env = {**os.environ}
    env["CLAUDE_CODE_USE_BEDROCK"] = "1"
    if BEDROCK_TOKEN:
        env["AWS_BEARER_TOKEN_BEDROCK"] = BEDROCK_TOKEN
    if AWS_REGION:
        env["AWS_REGION"] = AWS_REGION

    events = []
    final_text = ""
    stderr_lines = []

    start = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        # Read stderr in a background thread to avoid deadlocks
        def _read_stderr():
            for line in proc.stderr:
                stderr_lines.append(line)

        stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
        stderr_thread.start()

        # Stream stdout line by line (each line is a JSON event)
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue

            # Log raw JSON
            log_write(f"  STREAM: {line}")

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                print(f"  {C_DIM}[raw] {line}{C_RESET}")
                continue

            events.append(event)

            # Extract final text from result event
            if event.get("type") == "result":
                result_block = event.get("result", "")
                if isinstance(result_block, str):
                    final_text = result_block
                elif isinstance(result_block, dict):
                    # Sometimes result is nested
                    final_text = result_block.get("text", str(result_block))

            # Also collect text from assistant messages for validation
            if event.get("type") == "assistant":
                for block in event.get("message", {}).get("content", []):
                    if block.get("type") == "text":
                        final_text = block.get("text", "")

            # Format and print live
            formatted = format_stream_event(event)
            if formatted:
                print(formatted)
                sys.stdout.flush()

        proc.wait(timeout=TOOL_TIMEOUT)
        stderr_thread.join(timeout=5)
        elapsed = time.time() - start

        stderr_text = "".join(stderr_lines)

        return {
            "success": proc.returncode == 0,
            "output": final_text,
            "stderr": stderr_text,
            "returncode": proc.returncode,
            "elapsed": elapsed,
            "events": events,
        }

    except subprocess.TimeoutExpired:
        proc.kill()
        elapsed = time.time() - start
        return {
            "success": False,
            "output": final_text,
            "stderr": f"Timeout after {TOOL_TIMEOUT}s",
            "returncode": -1,
            "elapsed": elapsed,
            "events": events,
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "success": False,
            "output": "",
            "stderr": str(e),
            "returncode": -1,
            "elapsed": elapsed,
            "events": [],
        }


def run_tests(tool_filter=None):
    """
    Run Claude Code + leeroopedia-mcp production integration tests.

    Args:
        tool_filter: If set, only run tools whose name is in this set.
    """
    if not API_KEY:
        print("ERROR: LEEROOPEDIA_API_KEY not set. Add to .env or pass via CLI arg.")
        return False

    if not BEDROCK_TOKEN:
        print("ERROR: AWS_BEARER_TOKEN_BEDROCK not set. Required for Claude Code via Bedrock.")
        return False

    # Check Claude Code CLI
    claude_check = subprocess.run(["which", "claude"], capture_output=True)
    if claude_check.returncode != 0:
        print("ERROR: Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code")
        return False

    cases = TEST_CASES
    if tool_filter:
        cases = [(t, p, d) for t, p, d in TEST_CASES if t in tool_filter]

    if not cases:
        print(f"No matching tools for filter: {tool_filter}")
        return False

    # Create MCP config
    config_path = create_mcp_config(API_KEY)

    print("=" * 70)
    print(f"Leeroopedia MCP — Claude Code Production Test ({len(cases)} Tools)")
    print("=" * 70)
    print(f"MCP config:     {config_path}")
    print(f"API key:        {API_KEY[:12]}...{API_KEY[-4:]}")
    print(f"Bedrock:        CLAUDE_CODE_USE_BEDROCK=1 (region={AWS_REGION})")
    print(f"Allowed tools:  {ALLOWED_TOOLS}")
    print(f"Output format:  stream-json (live)")
    print("=" * 70)

    # Open log file
    log = open(LOG_PATH, "w", encoding="utf-8")

    def log_write(text: str):
        log.write(text + "\n")

    log_write("Leeroopedia MCP — Claude Code Production Test Results")
    log_write(f"Generated: {datetime.now().isoformat()}")
    log_write(f"API Key: {API_KEY[:12]}...{API_KEY[-4:]}")
    log_write(f"Bedrock: CLAUDE_CODE_USE_BEDROCK=1 (region={AWS_REGION})")
    log_write(f"MCP Config: {config_path}")
    log_write("=" * 80)

    # Log the MCP config
    log_write("\nMCP Config Content:")
    log_write(config_path.read_text())
    log_write("")

    results = []

    for i, (tool_name, prompt, description) in enumerate(cases, 1):
        print()
        print(f"{C_BOLD}{'=' * 70}{C_RESET}")
        print(
            f"{C_BOLD}[{i}/{len(cases)}] {tool_name}{C_RESET}: {description}"
        )
        print(f"{C_BOLD}{'=' * 70}{C_RESET}")

        # Log request
        log_write("")
        log_write("=" * 80)
        log_write(f"[{i}/{len(cases)}] {tool_name}")
        log_write(f"Description: {description}")
        log_write("-" * 80)
        log_write("PROMPT:")
        log_write(prompt)
        log_write("-" * 80)
        log_write("STREAM EVENTS:")

        print(f"  {C_DIM}Prompt: {prompt[:120]}...{C_RESET}")
        print(f"  {C_DIM}Streaming Claude Code output...{C_RESET}")
        print()

        result = run_claude_with_tool(prompt, config_path, tool_name, log_write)

        elapsed = result["elapsed"]
        output = result["output"]
        stderr = result["stderr"]
        events = result["events"]

        # Validate
        passed = True
        issues = []
        output_len = len(output.strip())
        output_lower = output.lower()

        if not result["success"]:
            passed = False
            issues.append(f"Claude Code exited with code {result['returncode']}")
            if stderr:
                # Only show non-MCP-server stderr (filter out noisy MCP logs)
                filtered = [
                    l for l in stderr.splitlines()
                    if "leeroopedia_mcp" not in l and "mcp" not in l.lower()
                ]
                if filtered:
                    issues.append(f"stderr: {' | '.join(filtered[:3])}")

        if output_len < MIN_RESPONSE_LENGTH:
            passed = False
            issues.append(
                f"Output too short ({output_len} chars, "
                f"min {MIN_RESPONSE_LENGTH})"
            )

        # Check for obvious error/failure patterns in output
        if "error" in output_lower and "tool" in output_lower and "failed" in output_lower:
            passed = False
            issues.append("Output contains tool failure indicators")

        # Check that Claude didn't say the tool doesn't exist
        if "does not exist" in output_lower or "not available" in output_lower:
            passed = False
            issues.append("Claude Code reports tool not available — check MCP server install")

        # Check that Claude didn't just refuse / ask for permission
        if output_len < 200 and ("permission" in output_lower or "grant" in output_lower):
            passed = False
            issues.append("Claude Code asked for permission instead of using the tool")

        # Check for API-level errors returned through the tool
        if "api error" in output_lower or "returned an api error" in output_lower:
            passed = False
            issues.append("MCP tool returned an API error — check api.leeroopedia.com connectivity")

        if "connection refused" in output_lower or "connection error" in output_lower:
            passed = False
            issues.append("MCP tool could not connect to API backend")

        status = "PASS" if passed else "FAIL"
        results.append((tool_name, status, elapsed, output_len, issues))

        # Count event types for summary
        event_counts = {}
        for e in events:
            et = e.get("type", "unknown")
            event_counts[et] = event_counts.get(et, 0) + 1

        # Log response
        log_write("-" * 80)
        log_write(f"FINAL TEXT OUTPUT ({output_len} chars):")
        log_write(output)
        if stderr:
            log_write("STDERR:")
            log_write(stderr)
        log_write("-" * 80)
        log_write(
            f"Status: {status} | Time: {elapsed:.1f}s | "
            f"Output: {output_len} chars | "
            f"Events: {event_counts} | "
            f"Return code: {result['returncode']}"
        )
        if issues:
            for issue in issues:
                log_write(f"Issue: {issue}")
        log_write("=" * 80)
        log.flush()

        # Console verdict
        color = C_GREEN if status == "PASS" else C_RED
        print()
        print(f"  {C_BOLD}Verdict:  {color}{status}{C_RESET}")
        print(f"  Output:   {output_len} chars")
        print(f"  Time:     {elapsed:.1f}s")
        print(f"  Events:   {event_counts}")
        if issues:
            for issue in issues:
                print(f"  {C_RED}Issue:    {issue}{C_RESET}")
        print()

    # Cleanup MCP config
    try:
        config_path.unlink()
    except OSError:
        pass

    # Summary
    summary_lines = []
    summary_lines.append("")
    summary_lines.append("=" * 70)
    summary_lines.append("SUMMARY")
    summary_lines.append("=" * 70)
    summary_lines.append(
        f"{'Tool':<32} {'Status':<8} {'Time':>7} {'Output':>8}"
    )
    summary_lines.append("-" * 70)

    total_time = 0
    pass_count = 0
    for tool_name, status, elapsed, length, issues in results:
        total_time += elapsed
        if status == "PASS":
            pass_count += 1
        summary_lines.append(
            f"{tool_name:<32} {status:<8} {elapsed:>6.1f}s {length:>7} chars"
        )
        for issue in issues:
            summary_lines.append(f"  └─ {issue}")

    summary_lines.append("-" * 70)
    summary_lines.append(
        f"{'Total':<32} {pass_count}/{len(results)} pass  {total_time:>6.1f}s"
    )
    summary_lines.append("=" * 70)

    for line in summary_lines:
        log_write(line)
    log.close()

    # Colored console summary
    print("=" * 70)
    print(f"{C_BOLD}SUMMARY{C_RESET}")
    print("=" * 70)
    print(f"{'Tool':<32} {'Status':<8} {'Time':>7} {'Output':>8}")
    print("-" * 70)
    for tool_name, status, elapsed, length, issues in results:
        color = C_GREEN if status == "PASS" else C_RED
        print(
            f"{tool_name:<32} {color}{status}{C_RESET}"
            f"{'':<{8-len(status)}} {elapsed:>6.1f}s {length:>7} chars"
        )
        for issue in issues:
            print(f"  └─ {issue}")
    print("-" * 70)
    print(f"{'Total':<32} {pass_count}/{len(results)} pass  {total_time:>6.1f}s")
    print("=" * 70)

    print(f"\nFull results log: {LOG_PATH}")

    if pass_count < len(results):
        print(
            f"\n{C_RED}FAILED: {len(results) - pass_count} tool(s) did not pass{C_RESET}"
        )
        return False
    else:
        print(f"\n{C_GREEN}ALL {len(results)} TOOLS PASSED{C_RESET}")
        return True


if __name__ == "__main__":
    # Args are tool name filters (e.g., get_page search_knowledge)
    tool_filter = set(sys.argv[1:]) if len(sys.argv) > 1 else None

    success = run_tests(tool_filter)
    sys.exit(0 if success else 1)
