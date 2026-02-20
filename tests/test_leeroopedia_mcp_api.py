# Test: Leeroopedia MCP API — All 8 Agentic Tools (End-to-End)
#
# Integration test that exercises the full Leeroopedia API stack:
#   Client → api.leeroopedia.com → Celery → Backend Search → AgenticKGSearch
#
# Tests the async task-based flow:
#   POST /v1/search  → {task_id, status: "queued"}
#   GET  /v1/search/task/{task_id}  → poll → {status: "success", results: ...}
#
# Requires:
#   - LEEROOPEDIA_API_KEY env var (or passed via CLI)
#   - api.leeroopedia.com reachable and connected to backend search service
#   - Neo4j + Weaviate running on the backend
#
# Run all tools:
#   python tests/test_leeroopedia_mcp_api.py
#
# Run only get_page (fast, no agent):
#   python tests/test_leeroopedia_mcp_api.py get_page
#
# Run specific tools:
#   python tests/test_leeroopedia_mcp_api.py get_page search_knowledge

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --- Configuration ---
API_URL = os.getenv("LEEROOPEDIA_API_URL", "https://api.leeroopedia.com")
API_KEY = os.getenv("LEEROOPEDIA_API_KEY", "")

# Polling config
POLL_INITIAL_INTERVAL = 0.5  # seconds
POLL_MAX_WAIT = 300  # 5 minutes per tool
POLL_MAX_DELAY = 5.0  # cap backoff at 5s
POLL_BACKOFF_FACTOR = 1.5

# Log file
LOG_PATH = Path(__file__).parent / "leeroopedia_mcp_api_results.log"

# Minimum response length
MIN_RESPONSE_LENGTH = 100

# Same test cases used throughout the test suite
TEST_CASES = [
    (
        "search_knowledge",
        {
            "query": "What is LoRA and how does it work?",
            "context": "Focus on parameter efficiency and rank selection",
        },
        "Research librarian synthesis with citations",
    ),
    (
        "build_plan",
        {
            "goal": "Fine-tune Llama-3 8B on a custom instruction dataset",
            "constraints": "Single A100 80GB GPU, complete within 4 hours",
        },
        "Step-by-step ML execution plan",
    ),
    (
        "review_plan",
        {
            "proposal": (
                "1. Load Llama-3 8B in 4-bit with QLoRA\n"
                "2. Use LoRA rank 64, alpha 128\n"
                "3. Train for 3 epochs with lr=2e-4\n"
                "4. Use batch size 4 with gradient accumulation 8\n"
                "5. Evaluate on held-out set"
            ),
            "goal": "Fine-tune Llama-3 8B for instruction following",
        },
        "Plan review with approvals, risks, suggestions",
    ),
    (
        "verify_code_math",
        {
            "code_snippet": (
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
            ),
            "concept_name": "LoRA low-rank adaptation",
        },
        "Code/math verification with Pass/Fail verdict",
    ),
    (
        "diagnose_failure",
        {
            "symptoms": "Training loss goes to NaN after ~100 steps during QLoRA fine-tuning of Llama-3 8B",
            "logs": (
                "Step 98: loss=0.853\n"
                "Step 99: loss=1.247\n"
                "Step 100: loss=nan\n"
                "Step 101: loss=nan\n"
                "RuntimeWarning: overflow encountered in float16"
            ),
        },
        "Failure diagnosis with fix and prevention",
    ),
    (
        "propose_hypothesis",
        {
            "current_status": (
                "Fine-tuned Llama-3 8B with QLoRA on instruction data. "
                "Training loss converged to 0.8 but eval performance is poor — "
                "model repeats itself and ignores instructions."
            ),
            "recent_experiments": (
                "Tried rank 16 and rank 64, both show same repetition. "
                "Increased dataset to 50k samples, no improvement."
            ),
        },
        "Ranked hypotheses with KB-grounded rationale",
    ),
    (
        "query_hyperparameter_priors",
        {
            "query": "Recommended learning rate, rank, and alpha for LoRA fine-tuning Llama-3 8B",
        },
        "Hyperparameter suggestion table with justification",
    ),
    (
        "get_page",
        {
            "page_id": "Heuristic/Huggingface_Alignment_handbook_QLoRA_Learning_Rate_Scaling",
        },
        "Direct page retrieval by exact ID (no agent)",
    ),
]


async def poll_task(client, task_id: str) -> dict:
    """
    Poll GET /v1/search/task/{task_id} until completion.

    Uses exponential backoff: 0.5s → 1.5x → capped at 5s.
    Max wait: 300 seconds.

    Returns:
        Response dict on success.

    Raises:
        TimeoutError: If task doesn't complete within max wait.
        RuntimeError: If task fails.
    """
    delay = POLL_INITIAL_INTERVAL
    start = time.monotonic()

    while time.monotonic() - start < POLL_MAX_WAIT:
        resp = await client.get(f"/v1/search/task/{task_id}")
        data = resp.json()
        status = data.get("status", "unknown")

        if status == "success":
            return data

        if status == "failure":
            error = data.get("error", "Unknown failure")
            raise RuntimeError(f"Task failed: {error}")

        elapsed = time.monotonic() - start
        logger.debug(f"Task {task_id} status={status} ({elapsed:.1f}s), retry in {delay:.1f}s")
        await asyncio.sleep(delay)
        delay = min(delay * POLL_BACKOFF_FACTOR, POLL_MAX_DELAY)

    raise TimeoutError(f"Task {task_id} did not complete within {POLL_MAX_WAIT}s")


async def run_tools(tool_filter=None):
    """
    Run Leeroopedia MCP API tests for selected tools.

    Sends real HTTP requests to api.leeroopedia.com, creates async tasks,
    and polls for results.

    Args:
        tool_filter: If set, only run tools whose name is in this set.
    """
    import httpx

    if not API_KEY:
        print("ERROR: LEEROOPEDIA_API_KEY not set. Pass via env or .env file.")
        return False

    cases = TEST_CASES
    if tool_filter:
        cases = [(t, a, d) for t, a, d in TEST_CASES if t in tool_filter]

    if not cases:
        print(f"No matching tools for filter: {tool_filter}")
        return False

    print("=" * 70)
    print(f"Leeroopedia MCP API — {len(cases)} Tool E2E Test")
    print(f"API: {API_URL}")
    print("=" * 70)

    log = open(LOG_PATH, "w", encoding="utf-8")

    def log_write(text: str):
        log.write(text + "\n")

    log_write("Leeroopedia MCP API E2E Test Results")
    log_write(f"Generated: {datetime.now().isoformat()}")
    log_write(f"API URL: {API_URL}")
    log_write(f"API Key: {API_KEY[:12]}...{API_KEY[-4:]}")
    log_write("=" * 80)

    results = []

    async with httpx.AsyncClient(
        base_url=API_URL,
        headers={
            "X-API-Key": API_KEY,
            "Content-Type": "application/json",
        },
        timeout=30.0,
    ) as client:

        # --- Step 0: Health / connectivity check ---
        print("\nChecking API connectivity...")
        try:
            resp = await client.get("/health")
            print(f"  /health: HTTP {resp.status_code} → {resp.text[:200]}")
            log_write(f"\nHealth check: HTTP {resp.status_code} → {resp.text[:200]}")
        except Exception as e:
            print(f"  /health: {e} (non-fatal, endpoint may not exist)")
            log_write(f"\nHealth check: {e}")
        log_write("")

        # --- Run each tool ---
        for i, (tool_name, arguments, description) in enumerate(cases, 1):
            print("-" * 70)
            print(f"[{i}/{len(cases)}] {tool_name}: {description}")
            print("-" * 70)

            payload = {"tool": tool_name, "arguments": arguments}

            # Log request
            log_write("")
            log_write("=" * 80)
            log_write(f"[{i}/{len(cases)}] {tool_name}")
            log_write(f"Description: {description}")
            log_write("-" * 80)
            log_write("REQUEST: POST /v1/search")
            log_write(json.dumps(payload, indent=2))
            log_write("-" * 80)

            start = time.time()
            try:
                # Step 1: Create task
                resp = await client.post("/v1/search", json=payload)
                create_data = resp.json()

                if resp.status_code != 200:
                    raise RuntimeError(
                        f"Task creation failed: HTTP {resp.status_code} → {resp.text[:300]}"
                    )

                task_id = create_data.get("task_id")
                if not task_id:
                    raise RuntimeError(f"No task_id in response: {create_data}")

                print(f"  Task created: {task_id} (status={create_data.get('status')})")
                log_write(f"Task created: {task_id}")
                log_write(f"Create response: {json.dumps(create_data)}")

                # Step 2: Poll for result
                print(f"  Polling...", end="", flush=True)
                result_data = await poll_task(client, task_id)
                elapsed = time.time() - start
                print(f" done ({elapsed:.1f}s)")

                # Validate response
                passed = True
                issues = []

                results_content = result_data.get("results", "")

                if not result_data.get("success", result_data.get("status") == "success"):
                    passed = False
                    issues.append(
                        f"success not true: {json.dumps(result_data)[:300]}"
                    )
                elif not isinstance(results_content, str):
                    passed = False
                    issues.append(f"results not a string: {type(results_content)}")
                elif len(results_content) < MIN_RESPONSE_LENGTH:
                    passed = False
                    issues.append(
                        f"Response too short ({len(results_content)} chars, "
                        f"min {MIN_RESPONSE_LENGTH})"
                    )

                latency_ms = result_data.get("latency_ms", 0)
                credits = result_data.get("credits_remaining")

                status = "PASS" if passed else "FAIL"
                content_len = len(results_content) if isinstance(results_content, str) else 0
                results.append((tool_name, status, elapsed, content_len, issues))

                # Log response
                log_write("RESPONSE (poll result):")
                log_write(f"status: {result_data.get('status')}")
                log_write(f"success: {result_data.get('success')}")
                log_write(f"latency_ms: {latency_ms}")
                log_write(f"credits_remaining: {credits}")
                log_write(f"results ({content_len} chars):")
                log_write(results_content if isinstance(results_content, str) else str(results_content))
                log_write("-" * 80)
                log_write(
                    f"Status: {status} | E2E Time: {elapsed:.1f}s | "
                    f"Length: {content_len} chars | "
                    f"API latency_ms: {latency_ms} | "
                    f"Credits: {credits}"
                )
                if issues:
                    for issue in issues:
                        log_write(f"Issue: {issue}")
                log_write("=" * 80)
                log.flush()

                # Console output
                print(f"\n  Status:       {status}")
                print(f"  Length:       {content_len} chars")
                print(f"  E2E Time:     {elapsed:.1f}s")
                print(f"  API latency:  {latency_ms}ms")
                print(f"  Credits:      {credits}")
                if issues:
                    for issue in issues:
                        print(f"  Issue:        {issue}")
                print(f"  Preview:      {results_content[:500]}...")
                print()

            except Exception as e:
                elapsed = time.time() - start
                results.append((tool_name, "ERROR", elapsed, 0, [str(e)]))

                log_write("RESPONSE:")
                log_write(f"ERROR: {e}")
                log_write(traceback.format_exc())
                log_write(f"Status: ERROR | Time: {elapsed:.1f}s")
                log_write("=" * 80)
                log.flush()

                print(f"\n  Status:  ERROR")
                print(f"  Time:    {elapsed:.1f}s")
                print(f"  Error:   {e}")
                traceback.print_exc()
                print()

    # Summary
    summary_lines = []
    summary_lines.append("")
    summary_lines.append("=" * 70)
    summary_lines.append("SUMMARY")
    summary_lines.append("=" * 70)
    summary_lines.append(
        f"{'Tool':<32} {'Status':<8} {'Time':>7} {'Length':>8}"
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
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Tool':<32} {'Status':<8} {'Time':>7} {'Length':>8}")
    print("-" * 70)
    for tool_name, status, elapsed, length, issues in results:
        color = "\033[32m" if status == "PASS" else "\033[31m"
        print(
            f"{tool_name:<32} {color}{status}\033[0m"
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
            f"\n\033[31mFAILED: {len(results) - pass_count} tool(s) did not pass\033[0m"
        )
        return False
    else:
        print(f"\n\033[32mALL {len(results)} TOOLS PASSED\033[0m")
        return True


if __name__ == "__main__":
    tool_filter = None
    args = sys.argv[1:]

    # Allow passing API key as first arg if it starts with kpsk_
    if args and args[0].startswith("kpsk_"):
        os.environ["LEEROOPEDIA_API_KEY"] = args[0]
        API_KEY_OVERRIDE = args[0]
        args = args[1:]
        # Update the module-level variable
        import __main__
        __main__.API_KEY = API_KEY_OVERRIDE

    if args:
        tool_filter = set(args)

    # Re-read API_KEY in case it was set via CLI arg
    api_key = os.getenv("LEEROOPEDIA_API_KEY", "")
    if not api_key:
        print("Usage: python tests/test_leeroopedia_mcp_api.py [API_KEY] [tool_name ...]")
        print("  Or set LEEROOPEDIA_API_KEY env var")
        sys.exit(1)

    success = asyncio.run(run_tools(tool_filter))
    sys.exit(0 if success else 1)
