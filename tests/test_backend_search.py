# Test: Backend Search Dispatch — All 8 Agentic Tools
#
# Integration test that exercises the backend search service layer
# (services/wiki_backend/backend_search/search.py) which dispatches
# tool invocations through execute_agentic_search() to AgenticKGSearch.
#
# Tests the async dispatch, SearchResult dataclass, error handling,
# and argument forwarding for all 8 tools.
#
# Requires:
#   - Neo4j and Weaviate Docker containers running
#   - data/wikis/.index present with indexed pages
#   - AWS Bedrock credentials (or ANTHROPIC_API_KEY)
#
# Run all tools:
#   python tests/test_backend_search.py
#
# Run only get_page (fast, no agent):
#   python tests/test_backend_search.py get_page
#
# Run a specific tool:
#   python tests/test_backend_search.py search_knowledge

import asyncio
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Add the backend_search package to sys.path so execute_agentic_search can be imported
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "wiki_backend"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Log file for full query/response pairs
LOG_PATH = Path(__file__).parent / "backend_search_results.log"

# Minimum response length to consider a tool result valid.
MIN_RESPONSE_LENGTH = 100

# Same test cases as test_agentic_kg_search.py, but formatted as
# (tool_name, arguments_dict, description) for execute_agentic_search().
TEST_CASES = [
    (
        "search_knowledge",
        {
            "query": "retrieval augmented generation"
        },
        "Research librarian synthesis with citations",
    )
]


async def run_tools(tool_filter=None):
    """
    Run backend search dispatch for selected tools.

    Args:
        tool_filter: If set, only run tools whose name is in this set.
                     If None, run all 8 tools.
    """
    from backend_search.search import execute_agentic_search, reset_backends

    cases = TEST_CASES
    if tool_filter:
        cases = [(t, a, d) for t, a, d in TEST_CASES if t in tool_filter]

    if not cases:
        print(f"No matching tools for filter: {tool_filter}")
        return False

    print("=" * 70)
    print(f"Backend Search Dispatch — {len(cases)} Tool Integration Test")
    print("=" * 70)

    # Open log file
    log = open(LOG_PATH, "w", encoding="utf-8")

    def log_write(text: str):
        log.write(text + "\n")

    log_write("Backend Search Dispatch Test Results")
    log_write(f"Generated: {datetime.now().isoformat()}")
    log_write("=" * 80)

    results = []

    for i, (tool_name, arguments, description) in enumerate(cases, 1):
        print("-" * 70)
        print(f"[{i}/{len(cases)}] {tool_name}: {description}")
        print("-" * 70)

        # Log query
        log_write("")
        log_write("=" * 80)
        log_write(f"[{i}/{len(cases)}] {tool_name}")
        log_write(f"Description: {description}")
        log_write("-" * 80)
        log_write("ARGUMENTS:")
        for key, value in arguments.items():
            log_write(f"  {key}:")
            for line in str(value).splitlines():
                log_write(f"    {line}")
        log_write("-" * 80)

        start = time.time()
        try:
            result = await execute_agentic_search(tool_name, arguments)
            elapsed = time.time() - start

            # Validate SearchResult
            passed = True
            issues = []

            if not result.success:
                passed = False
                issues.append(f"success=False, error={result.error}: {result.content[:200]}")
            elif not isinstance(result.content, str):
                passed = False
                issues.append(f"Expected str content, got {type(result.content).__name__}")
            elif result.content.startswith("Error:"):
                passed = False
                issues.append(f"Content starts with Error: {result.content[:200]}")
            elif len(result.content) < MIN_RESPONSE_LENGTH:
                passed = False
                issues.append(
                    f"Response too short ({len(result.content)} chars, "
                    f"min {MIN_RESPONSE_LENGTH})"
                )

            if result.latency_ms < 0:
                passed = False
                issues.append(f"Negative latency: {result.latency_ms}ms")

            status = "PASS" if passed else "FAIL"
            content_len = len(result.content) if isinstance(result.content, str) else 0
            results.append((tool_name, status, elapsed, content_len, issues))

            # Log full response
            log_write("RESPONSE:")
            log_write(result.content if isinstance(result.content, str) else str(result.content))
            log_write("-" * 80)
            log_write(
                f"Status: {status} | Time: {elapsed:.1f}s | "
                f"Length: {content_len} chars | "
                f"SearchResult.latency_ms: {result.latency_ms}"
            )
            if issues:
                for issue in issues:
                    log_write(f"Issue: {issue}")
            log_write("=" * 80)
            log.flush()

            # Console output
            print(f"\n  Status:       {status}")
            print(f"  Length:       {content_len} chars")
            print(f"  Time:         {elapsed:.1f}s")
            print(f"  latency_ms:   {result.latency_ms}")
            print(f"  success:      {result.success}")
            if result.error:
                print(f"  error:        {result.error}")
            if issues:
                for issue in issues:
                    print(f"  Issue:        {issue}")
            print(f"  Preview:      {result.content[:500]}...")
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

    # Cleanup singleton
    reset_backends()

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
        summary_lines.append(f"{tool_name:<32} {status:<8} {elapsed:>6.1f}s {length:>7} chars")
        for issue in issues:
            summary_lines.append(f"  └─ {issue}")

    summary_lines.append("-" * 70)
    summary_lines.append(f"{'Total':<32} {pass_count}/{len(results)} pass  {total_time:>6.1f}s")
    summary_lines.append("=" * 70)

    for line in summary_lines:
        log_write(line)
    log.close()

    # Print colored summary to console
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Tool':<32} {'Status':<8} {'Time':>7} {'Length':>8}")
    print("-" * 70)
    for tool_name, status, elapsed, length, issues in results:
        color = "\033[32m" if status == "PASS" else "\033[31m"
        print(f"{tool_name:<32} {color}{status}\033[0m{'':<{8-len(status)}} {elapsed:>6.1f}s {length:>7} chars")
        for issue in issues:
            print(f"  └─ {issue}")
    print("-" * 70)
    print(f"{'Total':<32} {pass_count}/{len(results)} pass  {total_time:>6.1f}s")
    print("=" * 70)

    print(f"\nFull results log: {LOG_PATH}")

    if pass_count < len(results):
        print(f"\n\033[31mFAILED: {len(results) - pass_count} tool(s) did not pass\033[0m")
        return False
    else:
        print(f"\n\033[32mALL {len(results)} TOOLS PASSED\033[0m")
        return True


# --- Error handling tests (always fast, no agent needed) ---

async def test_unknown_tool():
    """Test that an unknown tool returns a proper error SearchResult."""
    from backend_search.search import execute_agentic_search

    result = await execute_agentic_search("nonexistent_tool", {"query": "test"})
    assert not result.success, "Expected success=False for unknown tool"
    assert result.error == "unknown_tool", f"Expected error='unknown_tool', got '{result.error}'"
    assert "nonexistent_tool" in result.content
    print("  test_unknown_tool: PASS")


async def test_missing_argument():
    """Test that a missing required argument returns a proper error."""
    from backend_search.search import execute_agentic_search

    # search_knowledge requires 'query' — pass empty dict
    result = await execute_agentic_search("search_knowledge", {})
    assert not result.success, "Expected success=False for missing argument"
    assert result.error == "missing_argument", f"Expected error='missing_argument', got '{result.error}'"
    print("  test_missing_argument: PASS")


async def run_error_tests():
    """Run quick error-handling tests (no agent/DB needed)."""
    print("\n" + "=" * 70)
    print("Error Handling Tests")
    print("=" * 70)
    await test_unknown_tool()
    await test_missing_argument()
    print("=" * 70)
    print("All error tests passed")
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
