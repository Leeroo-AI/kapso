"""
E2E Test for RepoMemory - Seeded Repo Improvement

This test:
1. Seeds from a real repo (tests/fixtures/sample_repo_to_improve)
2. Runs Expert.build() to improve it
3. Dumps .praxium/repo_memory.json at baseline + after each experiment
4. Prints memory for human quality assessment

Prerequisites:
    - API keys in .env (OPENAI_API_KEY required, ANTHROPIC_API_KEY for claude_code)
    - Optional: ./start_infra.sh for KG support

Run:
    PYTHONPATH=. python tests/test_repo_memory_e2e.py
"""

import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

# Load env
from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# Test Configuration
# =============================================================================

SEED_REPO = Path(__file__).parent / "fixtures" / "sample_repo_to_improve"

GOAL = """Improve this data processing pipeline.

Current state:
- main.py loads CSV, computes stats, prints report
- stats.py computes mean, min, max, total
- report.py formats output

Requested improvements:
1. Add median and standard deviation to the statistics
2. Make main.py print "SCORE: X" where X is the number of stats computed (for evaluation)

Keep the existing structure. Only modify what's needed.
"""

TEST_CONFIG = {
    "goal": GOAL,
    "max_iterations": 2,
    "coding_agent": "openhands",  # Works without special deps
    "mode": "MINIMAL",
    "evaluator": "regex_pattern",
    "evaluator_params": {"pattern": r"SCORE:\s*(\d+)"},
}


# =============================================================================
# Helpers
# =============================================================================

def dump_repo_memory(workspace_dir: str, label: str) -> dict:
    """Load and print .praxium/repo_memory.json from a workspace."""
    memory_path = Path(workspace_dir) / ".praxium" / "repo_memory.json"
    
    print(f"\n{'='*70}")
    print(f"REPO MEMORY: {label}")
    print(f"{'='*70}")
    
    if not memory_path.exists():
        print("  [Not found]")
        return {}
    
    with open(memory_path) as f:
        doc = json.load(f)
    
    # Print summary
    repo_model = doc.get("repo_model", {})
    quality = doc.get("quality", {})
    experiments = doc.get("experiments", [])
    
    print(f"Generated at: {doc.get('generated_at', 'unknown')}")
    print(f"Summary: {repo_model.get('summary', '(none)')[:200]}")
    print(f"Claims: {len(repo_model.get('claims', []))}")
    print(f"Evidence OK: {quality.get('evidence_ok', False)}")
    print(f"Experiments recorded: {len(experiments)}")
    
    if repo_model.get("claims"):
        print("\nClaims:")
        for claim in repo_model.get("claims", [])[:5]:
            print(f"  - [{claim.get('kind')}] {claim.get('statement', '')[:80]}")
    
    if repo_model.get("entrypoints"):
        print("\nEntrypoints:")
        for ep in repo_model.get("entrypoints", [])[:3]:
            if isinstance(ep, dict):
                print(f"  - {ep.get('path')}: {ep.get('how_to_run', '')}")
            else:
                print(f"  - {ep}")
    
    if experiments:
        print("\nExperiment deltas:")
        for exp in experiments[-3:]:
            print(f"  - Branch: {exp.get('branch')}, files: {len(exp.get('changed_files', []))}")
    
    print(f"{'='*70}\n")
    return doc


def dump_branch_memory(repo, branch_name: str) -> dict:
    """Load repo memory from a specific branch without checkout."""
    from src.repo_memory import RepoMemoryManager
    
    doc = RepoMemoryManager.load_from_git_branch(repo, branch_name)
    if not doc:
        print(f"\n[Branch {branch_name}] No repo memory found")
        return {}
    
    print(f"\n{'='*70}")
    print(f"REPO MEMORY FROM BRANCH: {branch_name}")
    print(f"{'='*70}")
    
    repo_model = doc.get("repo_model", {})
    quality = doc.get("quality", {})
    experiments = doc.get("experiments", [])
    
    print(f"Summary: {repo_model.get('summary', '(none)')[:200]}")
    print(f"Claims: {len(repo_model.get('claims', []))}, Evidence OK: {quality.get('evidence_ok')}")
    print(f"Experiments recorded: {len(experiments)}")
    
    if experiments:
        last = experiments[-1]
        print(f"Last experiment: branch={last.get('branch')}, score={last.get('run_result', {}).get('score')}")
    
    print(f"{'='*70}\n")
    return doc


# =============================================================================
# Main Test
# =============================================================================

def run_test():
    """Run the E2E repo memory test."""
    from src.expert import Expert
    import git
    
    print("\n" + "=" * 70)
    print("REPO MEMORY E2E TEST")
    print("=" * 70)
    print(f"Seed repo: {SEED_REPO}")
    print(f"Goal: {TEST_CONFIG['goal'][:100]}...")
    print(f"Coding agent: {TEST_CONFIG['coding_agent']}")
    print(f"Max iterations: {TEST_CONFIG['max_iterations']}")
    print("=" * 70 + "\n")
    
    # Verify seed repo exists
    if not SEED_REPO.exists():
        print(f"ERROR: Seed repo not found at {SEED_REPO}")
        return False
    
    # Create output directory
    output_dir = tempfile.mkdtemp(prefix="repo_memory_e2e_")
    print(f"Output directory: {output_dir}\n")
    
    try:
        # Create Expert
        expert = Expert(domain="data_processing")
        
        # Run build with seed repo
        print("Starting Expert.build()...\n")
        solution = expert.build(
            goal=TEST_CONFIG["goal"],
            starting_repo_path=str(SEED_REPO),
            output_path=output_dir,
            max_iterations=TEST_CONFIG["max_iterations"],
            mode=TEST_CONFIG["mode"],
            coding_agent=TEST_CONFIG["coding_agent"],
            language="python",
            main_file="main.py",
            evaluator=TEST_CONFIG["evaluator"],
            evaluator_params=TEST_CONFIG["evaluator_params"],
        )
        
        # Get workspace path from solution
        workspace_dir = solution.code_path
        print(f"\n{'='*70}")
        print("BUILD COMPLETE")
        print(f"{'='*70}")
        print(f"Workspace: {workspace_dir}")
        print(f"Experiments: {len(solution.experiment_logs)}")
        print(f"Metadata: {solution.metadata}")
        
        # Dump baseline memory (from main branch)
        repo = git.Repo(workspace_dir)
        dump_branch_memory(repo, "main")
        
        # List all experiment branches and dump their memories
        branches = [ref.name for ref in repo.heads if ref.name != "main"]
        print(f"\nExperiment branches: {branches}")
        
        for branch in branches:
            dump_branch_memory(repo, branch)
        
        # Dump current worktree memory (should be best branch after checkout)
        dump_repo_memory(workspace_dir, "FINAL (best branch)")
        
        print("\n" + "=" * 70)
        print("TEST COMPLETE - Review memories above for quality")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        print(f"\nWorkspace kept at: {output_dir}")
        print("(Delete manually after review)")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys
    success = run_test()
    sys.exit(0 if success else 1)
