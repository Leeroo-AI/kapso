"""
E2E Test for RepoMemory - Seeded Repo Improvement (Claude Code + Bedrock)

This test:
1. Seeds from a real repo (tests/fixtures/sample_repo_to_improve)
2. Runs Kapso.evolve() to improve it using Claude Code with AWS Bedrock
3. Dumps .kapso/repo_memory.json at baseline + after each experiment
4. Prints memory for human quality assessment

Prerequisites:
    - AWS Bedrock credentials (AWS_BEARER_TOKEN_BEDROCK or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or AWS_PROFILE)
    - Claude Code CLI installed (`claude` command available)
    - OPENAI_API_KEY for RepoMemory inference (gpt-4o-mini)
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
# Bedrock Credential Check
# =============================================================================

def _has_bedrock_creds() -> bool:
    """Check if AWS Bedrock credentials are available."""
    return bool(
        os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        or (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))
        or os.environ.get("AWS_PROFILE")
    )


def _check_prerequisites() -> tuple[bool, str]:
    """Check all prerequisites for the test. Returns (ok, error_message)."""
    # Check Claude CLI
    if shutil.which("claude") is None:
        return False, "Claude Code CLI not installed (run: npm install -g @anthropic-ai/claude-code)"
    
    # Check Bedrock credentials
    if not _has_bedrock_creds():
        return False, (
            "No AWS Bedrock credentials found. Set one of:\n"
            "  - AWS_BEARER_TOKEN_BEDROCK\n"
            "  - AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY\n"
            "  - AWS_PROFILE"
        )
    
    # Check OpenAI key (for RepoMemory inference)
    if not os.environ.get("OPENAI_API_KEY"):
        return False, "OPENAI_API_KEY not set (required for RepoMemory inference with gpt-4o-mini)"
    
    return True, ""

from src.repo_memory.observation import (
    extract_repo_memory_sections_consulted,
)


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

# Claude Code + Bedrock configuration
# Uses Claude Opus 4.5 via AWS Bedrock for code generation
TEST_CONFIG = {
    "goal": GOAL,
    "max_iterations": 2,
    "coding_agent": "claude_code",  # Use Claude Code agent
    "mode": "MINIMAL",
    "evaluator": "regex_pattern",
    "evaluator_params": {"pattern": r"SCORE:\s*(\d+)"},
    # Bedrock-specific configuration for Claude Code
    "coding_agent_params": {
        "use_bedrock": True,
        "aws_region": os.environ.get("AWS_REGION", "us-east-1"),
        "streaming": True,
        "timeout": 180,  # Longer timeout for complex tasks
    },
    # Model configuration (Claude Opus 4.5 on Bedrock)
    "coding_agent_model": "us.anthropic.claude-opus-4-5-20251101-v1:0",
}


# =============================================================================
# Helpers
# =============================================================================

def _assert_repo_map_invariants(doc: dict, *, label: str) -> None:
    """
    Hard assertions for RepoMap portability + consistency.
    
    These invariants are critical because `.kapso/repo_memory.json` is committed into git
    experiment branches and must be portable across machines/runs.
    """
    repo_map = doc.get("repo_map", {}) or {}
    assert repo_map.get("repo_root") == ".", (
        f"{label}: expected repo_map.repo_root == '.', got {repo_map.get('repo_root')!r}"
    )

    files = repo_map.get("files", []) or []
    assert isinstance(files, list), f"{label}: expected repo_map.files to be a list"

    # Never include observability metadata or infrastructure paths in RepoMap.
    assert "changes.log" not in files, f"{label}: repo_map.files unexpectedly contains changes.log"
    assert not any(p.startswith(".kapso/") for p in files), f"{label}: repo_map.files contains .kapso/*"
    assert not any(p.startswith("sessions/") for p in files), f"{label}: repo_map.files contains sessions/*"


def _assert_changes_log_auditability(repo, *, branch_name: str, doc: dict) -> None:
    """
    Ensure observability is auditable:
    - `changes.log` is committed into the branch
    - it contains the explicit 'RepoMemory sections consulted:' line
    - parsed sections match persisted experiment metadata
    """
    try:
        changes_text = repo.git.show(f"{branch_name}:changes.log")
    except Exception as e:
        raise AssertionError(f"{branch_name}: changes.log not committed (git show failed): {e}") from e

    # This is the contract we instruct agents to follow.
    assert "repomemory sections consulted:" in changes_text.lower(), (
        f"{branch_name}: changes.log missing 'RepoMemory sections consulted:' line"
    )

    from_log = extract_repo_memory_sections_consulted(changes_text)

    experiments = doc.get("experiments", []) or []
    assert experiments, f"{branch_name}: expected experiments recorded in repo memory"
    last = experiments[-1] or {}
    assert last.get("branch") == branch_name, (
        f"{branch_name}: last experiment branch mismatch in repo memory: {last.get('branch')!r}"
    )

    rr = last.get("run_result", {}) or {}
    persisted = rr.get("repo_memory_sections_consulted", [])
    if not isinstance(persisted, list):
        persisted = []
    persisted = sorted(set(str(x) for x in persisted))

    assert persisted == from_log, (
        f"{branch_name}: repo_memory_sections_consulted mismatch\n"
        f"  from changes.log: {from_log}\n"
        f"  persisted:        {persisted}"
    )


def dump_repo_memory(workspace_dir: str, label: str) -> dict:
    """Load and print .kapso/repo_memory.json from a workspace."""
    memory_path = Path(workspace_dir) / ".kapso" / "repo_memory.json"
    
    print(f"\n{'='*70}")
    print(f"REPO MEMORY: {label}")
    print(f"{'='*70}")
    
    if not memory_path.exists():
        print("  [Not found]")
        return {}
    
    with open(memory_path) as f:
        doc = json.load(f)
    
    # Print summary (v2 book + legacy repo_model for compatibility)
    book = doc.get("book", {}) or {}
    repo_model = doc.get("repo_model", {})
    quality = doc.get("quality", {})
    experiments = doc.get("experiments", [])
    
    print(f"Generated at: {doc.get('generated_at', 'unknown')}")
    print(f"Schema: v{doc.get('schema_version')}")
    print(f"Book Summary: {(book.get('summary') or '(none)')[:200]}")
    print(f"Legacy Claims (flattened): {len(repo_model.get('claims', []))}")
    print(f"Evidence OK: {quality.get('evidence_ok', False)}")
    print(f"Experiments recorded: {len(experiments)}")
    
    # Show TOC + per-section claim counts (this is what agents use to navigate)
    toc = book.get("toc", []) or []
    sections = book.get("sections", {}) or {}
    if toc:
        print("\nBook TOC (claim counts):")
        for item in toc:
            sid = (item or {}).get("id", "")
            title = (item or {}).get("title", "")
            sec = sections.get(sid, {}) if isinstance(sections, dict) else {}
            claim_count = len((sec or {}).get("claims", []) or []) if isinstance((sec or {}).get("claims", []), list) else 0
            print(f"  - {sid}: {title} (claims={claim_count})")

    # High-signal semantic content (this is what matters for memory quality).
    # We print claim statements (not the whole JSON) so humans can quickly judge usefulness.
    if toc and isinstance(sections, dict):
        printed_any = False
        for item in toc:
            sid = (item or {}).get("id", "")
            sec = sections.get(sid, {}) if sid else {}
            claims = (sec or {}).get("claims", [])
            if not isinstance(claims, list) or not claims:
                continue
            if not printed_any:
                printed_any = True
                print("\nSemantic claims (first 5 per section):")
            print(f"  [{sid}]")
            for c in claims[:5]:
                stmt = (c or {}).get("statement", "")
                kind = (c or {}).get("kind", "?")
                print(f"    - [{kind}] {stmt}")
    
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
            rr = exp.get("run_result", {}) or {}
            consulted = rr.get("repo_memory_sections_consulted", [])
            ideation_consulted = rr.get("ideation_repo_memory_sections_consulted", [])
            print(
                f"  - Branch: {exp.get('branch')}, files: {len(exp.get('changed_files', []))}, "
                f"score: {rr.get('score')}, "
                f"ideation_repo_memory_sections_consulted: {ideation_consulted}, "
                f"repo_memory_sections_consulted: {consulted}"
            )
    
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
    
    book = doc.get("book", {}) or {}
    repo_model = doc.get("repo_model", {})
    quality = doc.get("quality", {})
    experiments = doc.get("experiments", [])
    
    print(f"Schema: v{doc.get('schema_version')}")
    print(f"Book Summary: {(book.get('summary') or '(none)')[:200]}")
    print(f"Legacy Claims (flattened): {len(repo_model.get('claims', []))}, Evidence OK: {quality.get('evidence_ok')}")
    print(f"Experiments recorded: {len(experiments)}")
    
    if experiments:
        last = experiments[-1]
        rr = last.get("run_result", {}) or {}
        print(
            f"Last experiment: branch={last.get('branch')}, score={rr.get('score')}, "
            f"ideation_repo_memory_sections_consulted={rr.get('ideation_repo_memory_sections_consulted', [])}, "
            f"repo_memory_sections_consulted={rr.get('repo_memory_sections_consulted', [])}"
        )
    
    print(f"{'='*70}\n")
    return doc


# =============================================================================
# Main Test
# =============================================================================

def _create_bedrock_config_file(output_dir: str) -> str:
    """Create a temporary config file with Claude Code + Bedrock settings."""
    import yaml
    
    config = {
        "default_mode": "BEDROCK_TEST",
        "modes": {
            "BEDROCK_TEST": {
                "search_strategy": {
                    "type": "llm_tree_search",
                    "params": {
                        "code_debug_tries": 2,
                        "idea_generation_model": "gpt-4o-mini",
                        "reasoning_effort": "medium",
                        "node_expansion_limit": 2,
                        "node_expansion_new_childs_count": 2,
                        "idea_generation_steps": 1,
                        "first_experiment_factor": 1,
                        "experimentation_per_run": 1,
                        "per_step_maximum_solution_count": 5,
                        "exploration_budget_percent": 40,
                        "idea_generation_ensemble_models": ["gpt-4o-mini"],
                    }
                },
                "coding_agent": {
                    "type": "claude_code",
                    "model": TEST_CONFIG["coding_agent_model"],
                    "debug_model": TEST_CONFIG["coding_agent_model"],
                    "agent_specific": TEST_CONFIG["coding_agent_params"],
                },
                "knowledge_search": {
                    "type": "kg_llm_navigation",
                    "enabled": False,
                },
                "evaluator": {
                    "type": "regex_pattern",
                    "params": TEST_CONFIG["evaluator_params"],
                },
                "stop_condition": {
                    "type": "never",
                    "params": {},
                },
            }
        }
    }
    
    config_path = os.path.join(output_dir, "bedrock_test_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path


def run_test():
    """Run the E2E repo memory test with Claude Code + Bedrock."""
    from src.execution.orchestrator import OrchestratorAgent
    from src.execution.coding_agents.factory import CodingAgentFactory
    from src.environment.handlers.generic import GenericProblemHandler
    import git
    
    print("\n" + "=" * 70)
    print("REPO MEMORY E2E TEST (Claude Code + Bedrock)")
    print("=" * 70)
    
    # Check prerequisites
    ok, err = _check_prerequisites()
    if not ok:
        print(f"ERROR: {err}")
        return False
    
    print(f"Seed repo: {SEED_REPO}")
    print(f"Goal: {TEST_CONFIG['goal'][:100]}...")
    print(f"Coding agent: {TEST_CONFIG['coding_agent']}")
    print(f"Model: {TEST_CONFIG['coding_agent_model']}")
    print(f"Max iterations: {TEST_CONFIG['max_iterations']}")
    print(f"Bedrock config: {TEST_CONFIG['coding_agent_params']}")
    print("=" * 70 + "\n")
    
    # Verify seed repo exists
    if not SEED_REPO.exists():
        print(f"ERROR: Seed repo not found at {SEED_REPO}")
        return False
    
    # Create output directory
    output_dir = tempfile.mkdtemp(prefix="repo_memory_e2e_bedrock_")
    workspace_dir = os.path.join(output_dir, "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Workspace directory: {workspace_dir}\n")
    
    try:
        # Create custom config file with Bedrock settings (in output_dir, not workspace)
        config_path = _create_bedrock_config_file(output_dir)
        print(f"Created config file: {config_path}\n")
        
        # Create problem handler
        handler = GenericProblemHandler(
            problem_description=TEST_CONFIG["goal"],
            main_file="main.py",
            language="python",
            timeout=300,
            evaluator=TEST_CONFIG["evaluator"],
            evaluator_params=TEST_CONFIG["evaluator_params"],
        )
        
        # Create orchestrator with custom config
        orchestrator = OrchestratorAgent(
            problem_handler=handler,
            config_path=config_path,
            mode="BEDROCK_TEST",
            coding_agent=None,  # Use config, not override
            is_kg_active=False,
            workspace_dir=workspace_dir,  # Use separate workspace subdirectory
            starting_repo_path=str(SEED_REPO),
        )
        
        # Verify the coding agent config was set correctly
        ca_config = orchestrator.search_strategy.workspace.coding_agent_config
        print(f"CodingAgentConfig:")
        print(f"  agent_type: {ca_config.agent_type}")
        print(f"  model: {ca_config.model}")
        print(f"  agent_specific: {ca_config.agent_specific}")
        print()
        
        # Run experimentation
        print("Starting experimentation loop...\n")
        orchestrator.solve(experiment_max_iter=TEST_CONFIG["max_iterations"])
        
        # Get workspace path (use the one from the strategy, which should match our workspace_dir)
        actual_workspace_dir = orchestrator.search_strategy.workspace.workspace_dir
        
        print(f"\n{'='*70}")
        print("BUILD COMPLETE")
        print(f"{'='*70}")
        print(f"Workspace: {actual_workspace_dir}")
        
        # Checkout to best solution
        orchestrator.search_strategy.checkout_to_best_experiment_branch()
        
        # Dump baseline memory (from main branch)
        repo = git.Repo(actual_workspace_dir)
        main_doc = dump_branch_memory(repo, "main")
        if main_doc:
            _assert_repo_map_invariants(main_doc, label="branch=main")
        
        # List all experiment branches and dump their memories
        branches = [ref.name for ref in repo.heads if ref.name != "main"]
        print(f"\nExperiment branches: {branches}")
        
        for branch in branches:
            doc = dump_branch_memory(repo, branch)
            if not doc:
                raise AssertionError(f"branch={branch}: expected repo memory to exist")
            _assert_repo_map_invariants(doc, label=f"branch={branch}")
            _assert_changes_log_auditability(repo, branch_name=branch, doc=doc)
        
        # Dump current worktree memory (should be best branch after checkout)
        final_doc = dump_repo_memory(actual_workspace_dir, "FINAL (best branch)")
        if final_doc:
            _assert_repo_map_invariants(final_doc, label="worktree=final")
        
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
