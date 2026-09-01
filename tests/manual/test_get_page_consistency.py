#!/usr/bin/env python3
"""
Run the get_page test case 10 times and verify all runs return the same page.

Uses the exact same approach as test_leeroopedia_mcp_claude_code.py:
  Claude Code CLI -> MCP stdio -> leeroopedia-mcp -> api.leeroopedia.com

Compares the RAW tool result (from MCP server) across all 10 runs,
not Claude's prose summary which varies due to LLM non-determinism.

Usage:
    conda activate kapso_conda
    python tests/test_get_page_consistency.py
"""

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

# Reuse the original test infrastructure
sys.path.insert(0, str(Path(__file__).parent))
from test_leeroopedia_mcp_claude_code import (
    create_mcp_config,
    run_claude_with_tool,
    API_KEY,
    BEDROCK_TOKEN,
    MCP_TOOL_PREFIX,
    LOG_PATH,
)

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# --- Configuration ---
NUM_RUNS = 10
PAGE_ID = "Heuristic/Huggingface_Alignment_handbook_DDP_Bias_Buffer_Ignore"

# ANSI colors
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_GREEN = "\033[32m"
C_RED = "\033[31m"
C_YELLOW = "\033[33m"
C_DIM = "\033[2m"


def extract_tool_result_from_events(events: list) -> str:
    """
    Extract the raw MCP tool result from Claude Code stream-json events.

    In Claude Code stream-json, the MCP tool result appears in a "user" type
    event that has a tool_use_result field containing the raw server response.
    """
    for event in events:
        if event.get("type") == "user":
            # tool_use_result is the raw MCP server response
            tur = event.get("tool_use_result", [])
            if tur:
                texts = []
                for block in tur:
                    if block.get("type") == "text":
                        texts.append(block.get("text", ""))
                if texts:
                    return "\n".join(texts)

            # Fallback: check message.content for tool_result blocks
            msg = event.get("message", {})
            content = msg.get("content", [])
            for block in content:
                if block.get("type") == "tool_result":
                    inner = block.get("content", [])
                    for inner_block in inner:
                        if inner_block.get("type") == "text":
                            return inner_block.get("text", "")
    return ""


def normalize_tool_result(text: str) -> str:
    """
    Strip the credits footer so we compare only actual page content.
    The credits line changes every call (counter decrements).
    """
    text = re.sub(r'\s*-{3}\s*\n\*Credits remaining: \d+\*\s*$', '', text).strip()
    return text


def main():
    if not API_KEY:
        print("ERROR: LEEROOPEDIA_API_KEY not set.")
        sys.exit(1)
    if not BEDROCK_TOKEN:
        print("ERROR: AWS_BEARER_TOKEN_BEDROCK not set.")
        sys.exit(1)

    config_path = create_mcp_config(API_KEY)

    # The prompt — same style as TEST_CASES in the original test
    prompt = (
        f"Use the {MCP_TOOL_PREFIX}get_page tool to retrieve the page with "
        f'page_id "{PAGE_ID}". '
        "Return the full page content you receive from the tool."
    )
    tool_name = "get_page"

    print("=" * 70)
    print(f"get_page Consistency Test — {NUM_RUNS} runs")
    print("=" * 70)
    print(f"Page ID:   {PAGE_ID}")
    print(f"MCP config: {config_path}")
    print(f"Comparing: RAW tool result from MCP server (not Claude's summary)")
    print("=" * 70)
    print()

    # Open log
    log = open(LOG_PATH, "w", encoding="utf-8")
    def log_write(text: str):
        log.write(text + "\n")

    raw_results = []    # raw tool result from MCP
    norm_results = []   # normalized (credits stripped)
    hashes = []
    rate_limited = []

    for i in range(1, NUM_RUNS + 1):
        print(f"{C_BOLD}--- Run {i}/{NUM_RUNS} ---{C_RESET}")
        sys.stdout.flush()

        # Use the exact same function as the original test
        result = run_claude_with_tool(prompt, config_path, tool_name, log_write)

        elapsed = result["elapsed"]
        events = result["events"]

        # Extract the RAW tool result from MCP (not Claude's prose)
        raw_tr = extract_tool_result_from_events(events)
        raw_results.append(raw_tr)

        # Check for rate limiting
        is_rl = "rate limit" in raw_tr.lower()
        rate_limited.append(is_rl)

        # Normalize: strip credits footer for comparison
        norm = normalize_tool_result(raw_tr)
        norm_results.append(norm)
        h = hashlib.md5(norm.encode()).hexdigest() if norm else "EMPTY"
        hashes.append(h)

        if is_rl:
            status = f"{C_YELLOW}RATE LIMITED{C_RESET}"
        elif not result["success"]:
            status = f"{C_RED}FAIL (exit code {result['returncode']}){C_RESET}"
        else:
            status = f"{C_GREEN}OK{C_RESET}"

        print(f"  Status:         {status}")
        print(f"  Time:           {elapsed:.1f}s")
        print(f"  Raw tool result: {len(raw_tr)} chars")
        print(f"  Normalized:     {len(norm)} chars (credits stripped)")
        print(f"  MD5:            {h}")

        # Preview of raw tool result
        preview = raw_tr[:300].replace("\n", "\\n") if raw_tr else "(empty)"
        print(f"  Raw preview:    {C_DIM}{preview}{C_RESET}")
        if norm:
            norm_preview = norm[:300].replace("\n", "\\n")
            print(f"  Norm preview:   {C_DIM}{norm_preview}{C_RESET}")
        print()
        sys.stdout.flush()

    log.close()

    # Cleanup MCP config
    try:
        config_path.unlink()
    except OSError:
        pass

    # --- Consistency analysis ---
    print("=" * 70)
    print(f"{C_BOLD}CONSISTENCY ANALYSIS (raw MCP tool result, credits stripped){C_RESET}")
    print("=" * 70)

    print(f"\n{'Run':<6} {'Hash':<34} {'Length':>8}  {'Note'}")
    print("-" * 70)
    for i, (h, norm, rl) in enumerate(zip(hashes, norm_results, rate_limited), 1):
        note = "RATE LIMITED" if rl else ""
        print(f"{i:<6} {h:<34} {len(norm):>8} chars  {note}")

    # Filter out rate-limited runs
    valid_hashes = [h for h, rl in zip(hashes, rate_limited) if not rl]
    valid_outputs = [n for n, rl in zip(norm_results, rate_limited) if not rl]
    rl_count = sum(rate_limited)

    print()
    if rl_count > 0:
        print(f"{C_YELLOW}NOTE: {rl_count} run(s) were rate-limited and excluded{C_RESET}")

    total_valid = len(valid_hashes)
    unique_valid = set(valid_hashes)

    if total_valid == 0:
        print(f"{C_RED}No valid runs to compare{C_RESET}")
        final_pass = False
    elif len(unique_valid) == 1:
        the_hash = valid_hashes[0]
        if the_hash == "EMPTY":
            print(f"{C_GREEN}{C_BOLD}ALL {total_valid} VALID RUNS CONSISTENT (empty page body){C_RESET}")
            print(f"  Page returned only credits footer — no actual content.")
        else:
            print(f"{C_GREEN}{C_BOLD}ALL {total_valid} VALID RUNS RETURNED THE SAME PAGE{C_RESET}")
            print(f"  Hash: {the_hash}")
            print(f"  Size: {len(valid_outputs[0])} chars")
        final_pass = True
    else:
        print(f"{C_RED}INCONSISTENCY: {len(unique_valid)} different results across {total_valid} valid runs{C_RESET}")
        for h in unique_valid:
            count = valid_hashes.count(h)
            idx = valid_hashes.index(h)
            preview = valid_outputs[idx][:300].replace("\n", "\\n") if valid_outputs[idx] else "(empty)"
            print(f"  Hash {h}: {count}x")
            print(f"    {C_DIM}{preview}{C_RESET}")
        final_pass = False

    print()
    print("=" * 70)
    if final_pass:
        print(f"{C_GREEN}{C_BOLD}RESULT: PASS — Consistent across {total_valid} valid runs{C_RESET}")
    else:
        print(f"{C_RED}{C_BOLD}RESULT: FAIL — Inconsistent across runs{C_RESET}")
    print("=" * 70)

    sys.exit(0 if final_pass else 1)


if __name__ == "__main__":
    main()
