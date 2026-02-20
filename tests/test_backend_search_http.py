# Test: Backend Search FastAPI HTTP — All 8 Agentic Tools
#
# Integration test that exercises the FastAPI HTTP layer
# (services/wiki_backend/backend_search/app.py) by sending
# POST /search requests through an in-process ASGI client.
#
# Uses httpx.AsyncClient with ASGITransport — no separate server
# process needed.
#
# Requires:
#   - Neo4j and Weaviate Docker containers running
#   - data/wikis/.index present with indexed pages
#   - AWS Bedrock credentials (or ANTHROPIC_API_KEY)
#
# Run all tools:
#   python tests/test_backend_search_http.py
#
# Run only get_page (fast, no agent):
#   python tests/test_backend_search_http.py get_page
#
# Run specific tools:
#   python tests/test_backend_search_http.py get_page search_knowledge

import asyncio
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Add the backend_search package to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "wiki_backend"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Log file for full query/response pairs
LOG_PATH = Path(__file__).parent / "backend_search_http_results.log"

# Minimum response length to consider a tool result valid.
MIN_RESPONSE_LENGTH = 100

# Same test cases as the other test files
TEST_CASES = [
    (
        "search_knowledge",
        {
            "query": "retrieval augmented generation",
        },
        "Research librarian synthesis with citations",
    ),
]


async def run_tools(tool_filter=None):
    """
    Run FastAPI HTTP tests for selected tools using in-process ASGI client.

    Args:
        tool_filter: If set, only run tools whose name is in this set.
                     If None, run all 8 tools.
    """
    import httpx
    from httpx import ASGITransport
    from backend_search.app import app

    cases = TEST_CASES
    if tool_filter:
        cases = [(t, a, d) for t, a, d in TEST_CASES if t in tool_filter]

    if not cases:
        print(f"No matching tools for filter: {tool_filter}")
        return False

    print("=" * 70)
    print(f"Backend Search FastAPI HTTP — {len(cases)} Tool Integration Test")
    print("=" * 70)

    log = open(LOG_PATH, "w", encoding="utf-8")

    def log_write(text: str):
        log.write(text + "\n")

    log_write("Backend Search FastAPI HTTP Test Results")
    log_write(f"Generated: {datetime.now().isoformat()}")
    log_write("=" * 80)

    results = []

    transport = ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

        # --- Health check first ---
        print("\nChecking /health endpoint...")
        resp = await client.get("/health")
        assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
        health = resp.json()
        assert health["status"] == "healthy"
        print(f"  /health: {health}")
        log_write(f"\nHealth check: {health}")
        log_write("")

        # --- Run each tool ---
        for i, (tool_name, arguments, description) in enumerate(cases, 1):
            print("-" * 70)
            print(f"[{i}/{len(cases)}] {tool_name}: {description}")
            print("-" * 70)

            # Build request payload
            payload = {"tool": tool_name, "arguments": arguments}

            # Log request
            log_write("")
            log_write("=" * 80)
            log_write(f"[{i}/{len(cases)}] {tool_name}")
            log_write(f"Description: {description}")
            log_write("-" * 80)
            log_write("REQUEST: POST /search")
            log_write(json.dumps(payload, indent=2))
            log_write("-" * 80)

            start = time.time()
            try:
                resp = await client.post(
                    "/search",
                    json=payload,
                    timeout=600.0,
                )
                elapsed = time.time() - start

                # Validate HTTP level
                passed = True
                issues = []

                if resp.status_code != 200:
                    passed = False
                    issues.append(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    results.append((tool_name, "FAIL", elapsed, 0, issues))

                    log_write(f"HTTP ERROR: {resp.status_code}")
                    log_write(resp.text[:1000])
                    log_write(f"Status: FAIL | Time: {elapsed:.1f}s")
                    log_write("=" * 80)
                    log.flush()

                    print(f"\n  Status:  FAIL")
                    print(f"  HTTP:    {resp.status_code}")
                    print(f"  Body:    {resp.text[:200]}")
                    print()
                    continue

                body = resp.json()

                # Validate SearchResponse schema
                for field in ("success", "content", "latency_ms"):
                    if field not in body:
                        passed = False
                        issues.append(f"Missing field: {field}")

                if not body.get("success"):
                    passed = False
                    issues.append(
                        f"success=false, error={body.get('error')}: "
                        f"{str(body.get('content', ''))[:200]}"
                    )
                elif not isinstance(body.get("content"), str):
                    passed = False
                    issues.append(
                        f"content not a string: {type(body.get('content'))}"
                    )
                elif body["content"].startswith("Error:"):
                    passed = False
                    issues.append(f"Content starts with Error: {body['content'][:200]}")
                elif len(body.get("content", "")) < MIN_RESPONSE_LENGTH:
                    passed = False
                    issues.append(
                        f"Response too short ({len(body['content'])} chars, "
                        f"min {MIN_RESPONSE_LENGTH})"
                    )

                if body.get("latency_ms", -1) < 0:
                    passed = False
                    issues.append(f"Negative latency_ms: {body.get('latency_ms')}")

                status = "PASS" if passed else "FAIL"
                content_len = len(body.get("content", ""))
                results.append((tool_name, status, elapsed, content_len, issues))

                # Log response
                log_write("RESPONSE: HTTP 200")
                log_write(f"success: {body.get('success')}")
                log_write(f"latency_ms: {body.get('latency_ms')}")
                log_write(f"error: {body.get('error')}")
                log_write(f"content ({content_len} chars):")
                log_write(body.get("content", ""))
                log_write("-" * 80)
                log_write(
                    f"Status: {status} | Time: {elapsed:.1f}s | "
                    f"Length: {content_len} chars | "
                    f"latency_ms: {body.get('latency_ms')}"
                )
                if issues:
                    for issue in issues:
                        log_write(f"Issue: {issue}")
                log_write("=" * 80)
                log.flush()

                # Console output
                print(f"\n  Status:       {status}")
                print(f"  HTTP:         {resp.status_code}")
                print(f"  Length:       {content_len} chars")
                print(f"  Time:         {elapsed:.1f}s")
                print(f"  latency_ms:   {body.get('latency_ms')}")
                print(f"  success:      {body.get('success')}")
                if body.get("error"):
                    print(f"  error:        {body['error']}")
                if issues:
                    for issue in issues:
                        print(f"  Issue:        {issue}")
                print(f"  Preview:      {body.get('content', '')[:500]}...")
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
    summary_lines.append(f"{'Tool':<32} {'Status':<8} {'Time':>7} {'Length':>8}")
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


# --- Error handling tests (fast, no agent needed) ---

async def run_error_tests():
    """Run quick HTTP-level error handling tests."""
    import httpx
    from httpx import ASGITransport
    from backend_search.app import app

    print("\n" + "=" * 70)
    print("HTTP Error Handling Tests")
    print("=" * 70)

    transport = ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

        # 1. Unknown tool
        resp = await client.post(
            "/search",
            json={"tool": "nonexistent_tool", "arguments": {"query": "test"}},
        )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        body = resp.json()
        assert body["success"] is False
        assert body["error"] == "unknown_tool"
        assert "nonexistent_tool" in body["content"]
        print("  test_unknown_tool: PASS")

        # 2. Missing required argument
        resp = await client.post(
            "/search",
            json={"tool": "search_knowledge", "arguments": {}},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert body["error"] == "missing_argument"
        print("  test_missing_argument: PASS")

        # 3. Invalid request body (missing 'tool' field)
        resp = await client.post(
            "/search",
            json={"arguments": {"query": "test"}},
        )
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"
        print("  test_invalid_request_body: PASS")

        # 4. Health endpoint
        resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["service"] == "backend_search"
        print("  test_health_endpoint: PASS")

    print("=" * 70)
    print("All HTTP error tests passed")
    print("=" * 70)


if __name__ == "__main__":
    tool_filter = None
    if len(sys.argv) > 1:
        tool_filter = set(sys.argv[1:])

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Always run error handling tests first (fast)
        loop.run_until_complete(run_error_tests())

        # Then run tool dispatch tests
        success = loop.run_until_complete(run_tools(tool_filter))
        sys.exit(0 if success else 1)
    finally:
        loop.close()
