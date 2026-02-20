# Test: Agentic KG Search — All 8 Tools
#
# Integration test that exercises every AgenticKGSearch tool method
# against the live KG (Neo4j + Weaviate must be running).
#
# Writes a detailed log file (tests/agentic_search_results.log) with
# the full query inputs and complete agent responses for each tool.
#
# Requires:
#   - Neo4j and Weaviate Docker containers running
#   - data/wikis/.index present with indexed pages
#   - AWS Bedrock credentials (or ANTHROPIC_API_KEY)
#
# Run:
#   python tests/test_agentic_kg_search.py

import logging
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

# Log file for full query/response pairs
LOG_PATH = Path(__file__).parent / "agentic_search_results.log"

# Minimum response length to consider a tool result valid.
MIN_RESPONSE_LENGTH = 100


def run_all_tools():
    """Run all 8 agentic search tools and report results."""

    from kapso.knowledge_base.search.agentic_kg_search import AgenticKGSearch

    print("=" * 70)
    print("Agentic KG Search — 8 Tool Integration Test")
    print("=" * 70)

    # Open log file
    log = open(LOG_PATH, "w", encoding="utf-8")

    def log_write(text: str):
        """Write to both stdout and log file."""
        log.write(text + "\n")

    log_write(f"Agentic KG Search Test Results")
    log_write(f"Generated: {datetime.now().isoformat()}")
    log_write("=" * 80)

    # Initialize once — agent is reused across all calls
    print("\nInitializing AgenticKGSearch (from knowledge_search.yaml)...")
    t0 = time.time()
    search = AgenticKGSearch()
    init_time = time.time() - t0
    print(f"Initialized in {init_time:.1f}s\n")
    log_write(f"\nAgent initialized in {init_time:.1f}s")
    log_write("")

    # Define test cases: (tool_name, kwargs_dict, description)
    # kwargs are stored explicitly so we can log them.
    test_cases = [
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

    results = []

    for i, (tool_name, kwargs, description) in enumerate(test_cases, 1):
        print("-" * 70)
        print(f"[{i}/{len(test_cases)}] {tool_name}: {description}")
        print("-" * 70)

        # Log query
        log_write("")
        log_write("=" * 80)
        log_write(f"[{i}/{len(test_cases)}] {tool_name}")
        log_write(f"Description: {description}")
        log_write("-" * 80)
        log_write("QUERY:")
        for key, value in kwargs.items():
            log_write(f"  {key}:")
            for line in str(value).splitlines():
                log_write(f"    {line}")
        log_write("-" * 80)

        start = time.time()
        try:
            method = getattr(search, tool_name)
            output = method(**kwargs)
            elapsed = time.time() - start

            # Validation checks
            passed = True
            issues = []

            if not isinstance(output, str):
                passed = False
                issues.append(f"Expected str, got {type(output).__name__}")
            elif output.startswith("Error:"):
                passed = False
                issues.append(f"Agent returned error: {output[:200]}")
            elif len(output) < MIN_RESPONSE_LENGTH:
                passed = False
                issues.append(
                    f"Response too short ({len(output)} chars, "
                    f"min {MIN_RESPONSE_LENGTH})"
                )

            status = "PASS" if passed else "FAIL"
            results.append((tool_name, status, elapsed, len(output), issues))

            # Log full response
            log_write("RESPONSE:")
            log_write(output if isinstance(output, str) else str(output))
            log_write("-" * 80)
            log_write(f"Status: {status} | Time: {elapsed:.1f}s | Length: {len(output)} chars")
            if issues:
                for issue in issues:
                    log_write(f"Issue: {issue}")
            log_write("=" * 80)
            log.flush()

            # Console output
            print(f"\n  Status:  {status}")
            print(f"  Length:  {len(output)} chars")
            print(f"  Time:    {elapsed:.1f}s")
            if issues:
                for issue in issues:
                    print(f"  Issue:   {issue}")
            print(f"  Preview: {output[:500]}...")
            print()

        except Exception as e:
            elapsed = time.time() - start
            results.append((tool_name, "ERROR", elapsed, 0, [str(e)]))

            # Log error
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

    # Cleanup
    search.close()

    # Summary (both console and log)
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

    # Write plain summary to log
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


if __name__ == "__main__":
    success = run_all_tools()
    sys.exit(0 if success else 1)
