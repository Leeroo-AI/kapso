#!/usr/bin/env python3
"""
Merge Simulation Test — Two-Round Sequential Merge

Simulates the full KnowledgePipeline merge process by bypassing ingestion
and reading pre-existing staging pages directly. Runs two rounds:

  Round 1 (Alibaba_ROLL):          No .index → "Create All" mode
  Round 2 (Allenai_Open_instruct): .index exists → "Agentic Merge" mode

This exercises both merge code paths end-to-end.

Usage:
    conda activate kapso_conda
    python tests/test_merge_simulation.py
"""

import shutil
import sys
import time
from collections import Counter
from pathlib import Path

# Add src/ to path so we can import kapso modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kapso.knowledge_base.search.kg_graph_search import parse_wiki_directory
from kapso.knowledge_base.learners.merger import KnowledgeMerger
from kapso.core.config import load_config

# =============================================================================
# Configuration
# =============================================================================

# Staging directories (output of repo ingestion, already completed)
STAGING_SOURCES = [
    ("Alibaba_ROLL",          Path("data/wikis/_staging/Alibaba_ROLL/0ea299b52a58")),
    ("Allenai_Open_instruct", Path("data/wikis/_staging/Allenai_Open_instruct/624747879f99")),
]

# Target wiki directory for the merge test
WIKI_DIR = Path("data/wikis_merge_test")


# =============================================================================
# Helpers
# =============================================================================

def print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}")


def print_page_summary(pages, label: str) -> None:
    """Print a breakdown of pages by type."""
    type_counts = Counter(p.page_type for p in pages)
    print(f"\n  {label}: {len(pages)} pages")
    for ptype in ["Workflow", "Principle", "Implementation", "Heuristic", "Environment"]:
        count = type_counts.get(ptype, 0)
        if count > 0:
            print(f"    {ptype:<20s} {count}")


def print_merge_result(result, round_name: str) -> None:
    """Print merge result details."""
    print(f"\n  {round_name} Result:")
    print(f"    Created:  {len(result.created)}")
    print(f"    Edited:   {len(result.edited)}")
    print(f"    Failed:   {len(result.failed)}")
    print(f"    Errors:   {len(result.errors)}")
    if result.plan_path:
        print(f"    Plan:     {result.plan_path}")
    if result.errors:
        print(f"\n    Error details:")
        for err in result.errors[:5]:
            print(f"      - {err}")
        if len(result.errors) > 5:
            print(f"      ... and {len(result.errors) - 5} more")


def count_wiki_files(wiki_dir: Path) -> dict:
    """Count .md files per type subdirectory."""
    counts = {}
    for subdir in ["workflows", "principles", "implementations", "heuristics", "environments"]:
        path = wiki_dir / subdir
        if path.exists():
            counts[subdir] = len(list(path.glob("*.md")))
        else:
            counts[subdir] = 0
    return counts


# =============================================================================
# Main
# =============================================================================

def main():
    print_header("Merge Simulation — Two-Round Sequential Test")
    print(f"\n  Target wiki_dir: {WIKI_DIR}")
    print(f"  Sources: {len(STAGING_SOURCES)}")
    for name, path in STAGING_SOURCES:
        print(f"    - {name}: {path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 0: Clean start — remove wiki_dir if it exists
    # ─────────────────────────────────────────────────────────────────────────
    if WIKI_DIR.exists():
        print(f"\n  Cleaning up existing {WIKI_DIR}...")
        shutil.rmtree(WIKI_DIR)
        print(f"  Removed.")

    # ─────────────────────────────────────────────────────────────────────────
    # Process each source sequentially (Round 1, then Round 2)
    # ─────────────────────────────────────────────────────────────────────────
    for round_num, (source_name, staging_dir) in enumerate(STAGING_SOURCES, 1):
        print_header(f"Round {round_num} — {source_name}")

        # Check staging dir exists
        if not staging_dir.exists():
            print(f"  ERROR: Staging directory not found: {staging_dir}")
            sys.exit(1)

        # ─────────────────────────────────────────────────────────────────────
        # Stage 1 (bypass): Parse pages from staging directory
        # ─────────────────────────────────────────────────────────────────────
        print(f"\n  Parsing pages from {staging_dir}...")
        pages = parse_wiki_directory(staging_dir)
        print_page_summary(pages, f"Parsed from {source_name}")

        # ─────────────────────────────────────────────────────────────────────
        # Stage 2: Merge — same call as KnowledgePipeline.run() line 223
        # ─────────────────────────────────────────────────────────────────────
        # Re-instantiate merger each round so _try_initialize_index() picks up
        # any .index created in the previous round.
        #
        # Load merger config from config.yaml (same as Kapso.learn() does):
        #   use_bedrock, aws_region, model, timeout, etc.
        index_path = WIKI_DIR / ".index"
        has_index = index_path.exists()
        mode = "Agentic Merge" if has_index else "Create All"
        print(f"\n  .index exists: {has_index}")
        print(f"  Merge mode:    {mode}")
        print(f"\n  Starting merge...")

        # Build merger params from config.yaml (same as Kapso.learn() lines 430-452)
        config_path = Path(__file__).resolve().parent.parent / "src" / "kapso" / "config.yaml"
        config = load_config(str(config_path))
        mode_name = config.get("default_mode", "GENERIC")
        mode_config = config.get("modes", {}).get(mode_name, {})
        learner_config = mode_config.get("learner", {})
        merger_params = learner_config.get("merger", {}).copy()

        # If .index exists, pass it so MCP server can initialize the backend
        if has_index:
            merger_params["kg_index_path"] = str(index_path.resolve())

        merger = KnowledgeMerger(agent_config=merger_params)
        start_time = time.time()

        try:
            # Pass staging_dir so the agentic merge prompt can reference
            # candidate page files on disk instead of inline content
            result = merger.merge(pages, wiki_dir=WIKI_DIR, staging_dir=staging_dir)
        except Exception as e:
            print(f"\n  MERGE FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue
        finally:
            elapsed = time.time() - start_time
            print(f"\n  Elapsed: {elapsed:.1f}s")

        # Print results
        print_merge_result(result, f"Round {round_num}")

        # Show wiki_dir file counts after this round
        file_counts = count_wiki_files(WIKI_DIR)
        total_files = sum(file_counts.values())
        print(f"\n  Files in {WIKI_DIR}: {total_files}")
        for subdir, count in file_counts.items():
            if count > 0:
                print(f"    {subdir:<20s} {count}")

    # ─────────────────────────────────────────────────────────────────────────
    # Final Verification
    # ─────────────────────────────────────────────────────────────────────────
    print_header("Final Verification")

    # Count final wiki files
    file_counts = count_wiki_files(WIKI_DIR)
    total_files = sum(file_counts.values())
    print(f"\n  Total .md files in {WIKI_DIR}: {total_files}")
    for subdir, count in file_counts.items():
        if count > 0:
            print(f"    {subdir:<20s} {count}")

    # Check .index file
    index_path = WIKI_DIR / ".index"
    if index_path.exists():
        import json
        index_data = json.loads(index_path.read_text())
        print(f"\n  .index file: {index_path}")
        print(f"    backend:    {index_data.get('search_backend', '?')}")
        print(f"    page_count: {index_data.get('page_count', '?')}")
        print(f"    created_at: {index_data.get('created_at', '?')}")
    else:
        print(f"\n  WARNING: No .index file found at {index_path}")

    # Check _merge_plan.md (written by agentic merge in Round 2 to staging_dir)
    # The plan is written to the last staging directory used
    last_staging = STAGING_SOURCES[-1][1] if STAGING_SOURCES else None
    merge_plan = last_staging / "_merge_plan.md" if last_staging else None
    if merge_plan and merge_plan.exists():
        content = merge_plan.read_text()
        lines = content.strip().split("\n")
        print(f"\n  _merge_plan.md: {merge_plan} ({len(lines)} lines)")
        # Show first few lines as preview
        for line in lines[:10]:
            print(f"    {line}")
        if len(lines) > 10:
            print(f"    ... ({len(lines) - 10} more lines)")
    else:
        print(f"\n  No _merge_plan.md (expected if Round 2 didn't run agentic merge)")

    print(f"\n{'=' * 70}")
    print("  Merge simulation complete!")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
