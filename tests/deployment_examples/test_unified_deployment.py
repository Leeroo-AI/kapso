#!/usr/bin/env python3
"""
Unified Deployment Test

Tests the Expert deployment flow as shown in the README:

    from src.expert import Expert, DeployStrategy
    
    expert = Expert(domain="testing")
    solution = expert.build(goal="...", output_path="./repo")  # We create SolutionResult manually
    software = expert.deploy(solution, strategy=DeployStrategy.LOCAL)
    result = software.run({"input": "data"})
    software.stop()

Each repo in input_repos/ represents a "built solution" that we pass to deploy().

Usage:
    python test_unified_deployment.py              # Run all repos with LOCAL strategy
    python test_unified_deployment.py sentiment    # Run specific repo
    python test_unified_deployment.py --strategy local  # Force specific strategy
    python test_unified_deployment.py --strategy auto   # Use AUTO selection

Requirements:
    - ANTHROPIC_API_KEY in .env (for AUTO strategy and adaptation)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
EXAMPLES_DIR = Path(__file__).parent
INPUT_DIR = EXAMPLES_DIR / "input_repos"

# Load environment
load_dotenv(PROJECT_ROOT / ".env")

# Add project to path
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# ANSI Colors
# =============================================================================
class C:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def header(text: str):
    print(f"\n{C.BOLD}{C.BLUE}{'='*70}")
    print(f"{text}")
    print(f"{'='*70}{C.END}\n")


def subheader(text: str):
    print(f"\n{C.BOLD}{C.CYAN}--- {text} ---{C.END}\n")


def success(text: str):
    print(f"{C.GREEN}✓ {text}{C.END}")


def error(text: str):
    print(f"{C.RED}✗ {text}{C.END}")


def info(text: str):
    print(f"{C.BLUE}ℹ {text}{C.END}")


def warn(text: str):
    print(f"{C.YELLOW}⚠ {text}{C.END}")


# =============================================================================
# Test Configuration
# =============================================================================
@dataclass
class RepoConfig:
    """Configuration for a test repository (simulates a built solution)."""
    name: str
    goal: str
    sample_input: Dict[str, Any]


# All test repositories - each represents a "built solution"
REPOS: List[RepoConfig] = [
    RepoConfig(
        name="sentiment",
        goal="Sentiment analysis API using TextBlob",
        sample_input={"text": "I love this product!"},
    ),
    RepoConfig(
        name="image",
        goal="Image processing REST API with Pillow",
        # Sample 1x1 red PNG image (base64 encoded)
        # The processor expects image_data, not operation
        sample_input={
            "image_data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==",
            "width": 100,
        },
    ),
    RepoConfig(
        name="embeddings",
        goal="Text embeddings API using sentence-transformers with GPU",
        sample_input={"text": "Hello world"},
    ),
    RepoConfig(
        name="qa",
        goal="Question answering using transformers pipeline",
        sample_input={"question": "What is AI?", "context": "AI is artificial intelligence."},
    ),
    RepoConfig(
        name="classifier",
        goal="Production ML text classification with batching",
        sample_input={"text": "This is a great product!"},
    ),
    RepoConfig(
        name="chatbot",
        goal="Conversational AI chatbot using LangGraph with memory",
        sample_input={"text": "Hello! Who are you?"},
    ),
]


# =============================================================================
# Test Result Tracking
# =============================================================================
@dataclass
class TestResult:
    """Result of testing a single repository."""
    name: str
    strategy: str
    deploy_ok: bool
    run_ok: bool
    output: Optional[Any]
    error: Optional[str]


# =============================================================================
# Main runner function - Following README User Flow
# =============================================================================
#
# NOTE:
# This module is primarily intended to be run as a script:
#   PYTHONPATH=. python tests/deployment_examples/test_unified_deployment.py --strategy local
#
# We intentionally do NOT expose this as a pytest test (no `test_*` prefix)
# because it depends on external runtime deployment capabilities.
def run_repo(repo: RepoConfig, strategy: str = "local") -> TestResult:
    """
    Test a single repository following the README user flow:
    
        expert = Expert(domain="testing")
        software = expert.deploy(solution, strategy=DeployStrategy.LOCAL)
        result = software.run(sample_input)
        software.stop()
    """
    from src.expert import Expert, DeployStrategy
    from src.execution.solution import SolutionResult
    
    subheader(f"Testing: {repo.name}")
    
    result = TestResult(
        name=repo.name,
        strategy=strategy,
        deploy_ok=False,
        run_ok=False,
        output=None,
        error=None,
    )
    
    input_path = INPUT_DIR / repo.name
    
    if not input_path.exists():
        result.error = f"Input repo not found: {input_path}"
        error(result.error)
        return result
    
    info(f"Repo path: {input_path}")
    info(f"Goal: {repo.goal}")
    
    try:
        # =====================================================================
        # Step 1: Create Expert (simulating user initialization)
        # =====================================================================
        expert = Expert(domain="testing")
        info("Created Expert(domain='testing')")
        
        # =====================================================================
        # Step 2: Create SolutionResult (simulating expert.build() output)
        # In real usage, this comes from expert.build()
        # Here we create it manually from our test repos
        # =====================================================================
        solution = SolutionResult(
            goal=repo.goal,
            code_path=str(input_path),
        )
        info(f"Created SolutionResult(goal='{repo.goal[:40]}...', code_path='{input_path}')")
        
        # =====================================================================
        # Step 3: Deploy the solution
        # expert.deploy(solution, strategy=DeployStrategy.LOCAL)
        # =====================================================================
        strategy_upper = strategy.upper()
        try:
            deploy_strategy = DeployStrategy[strategy_upper]
        except KeyError:
            available = [s.name for s in DeployStrategy]
            result.error = f"Unknown strategy '{strategy}'. Available: {available}"
            error(result.error)
            return result
        
        import time
        deploy_start = time.time()
        info(f"Calling expert.deploy(solution, strategy=DeployStrategy.{strategy_upper})...")
        
        software = expert.deploy(solution, strategy=deploy_strategy)
        
        deploy_elapsed = time.time() - deploy_start
        result.strategy = software.name
        result.deploy_ok = True
        
        success(f"software = expert.deploy() completed in {deploy_elapsed:.1f}s")
        info(f"  software.name: {software.name}")
        info(f"  software.is_healthy(): {software.is_healthy()}")
        
        # =====================================================================
        # Step 4: Run the software
        # result = software.run(sample_input)
        # =====================================================================
        info(f"Calling software.run({repo.sample_input})...")
        
        run_start = time.time()
        response = software.run(repo.sample_input)
        run_elapsed = time.time() - run_start
        
        result.output = response
        
        if response.get("status") == "success":
            result.run_ok = True
            success(f"result = software.run() completed in {run_elapsed:.1f}s")
            output_preview = str(response.get("output", {}))[:80]
            info(f"  result['status']: success")
            info(f"  result['output']: {output_preview}...")
        else:
            result.run_ok = False
            warn(f"software.run() returned error: {response.get('error', 'Unknown')}")
        
        # =====================================================================
        # Step 5: Stop the software
        # software.stop()
        # =====================================================================
        software.stop()
        info("software.stop() called")
        
    except Exception as e:
        result.error = str(e)
        error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def print_summary(results: List[TestResult]):
    """Print test summary."""
    header("SUMMARY")
    
    print(f"{'Repo':<12} {'Strategy':<12} {'Deploy':<10} {'Run':<10} {'Error':<30}")
    print("-" * 80)
    
    for r in results:
        deploy = f"{C.GREEN}✓{C.END}" if r.deploy_ok else f"{C.RED}✗{C.END}"
        run = f"{C.GREEN}✓{C.END}" if r.run_ok else f"{C.RED}✗{C.END}"
        err = (r.error[:27] + "...") if r.error and len(r.error) > 30 else (r.error or "-")
        
        print(f"{r.name:<12} {r.strategy:<12} {deploy:<19} {run:<19} {err:<30}")
    
    print()
    
    # Stats
    total = len(results)
    deploy_ok = sum(1 for r in results if r.deploy_ok)
    run_ok = sum(1 for r in results if r.run_ok)
    
    print(f"Deploy success: {deploy_ok}/{total}")
    print(f"Run success: {run_ok}/{total}")
    print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Deployment Test")
    parser.add_argument("repos", nargs="*", help="Specific repos to test (default: all)")
    parser.add_argument("--strategy", "-s", default="local", 
                        help="Deployment strategy (auto, local, docker, modal, bentoml, langgraph)")
    args = parser.parse_args()
    
    header("UNIFIED DEPLOYMENT TEST")
    print("Testing the Expert deployment flow from README:\n")
    print("    expert = Expert(domain='testing')")
    print("    solution = SolutionResult(goal='...', code_path='./repo')")
    print("    software = expert.deploy(solution, strategy=DeployStrategy.LOCAL)")
    print("    result = software.run(sample_input)")
    print("    software.stop()")
    print()
    
    # Check credentials
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        warn("ANTHROPIC_API_KEY not set - AUTO strategy and adaptation may fail")
    else:
        info(f"ANTHROPIC_API_KEY: {anthropic_key[:20]}...")
    
    print(f"\nInput repos: {INPUT_DIR}")
    print(f"Strategy: {args.strategy}")
    print()
    
    # Filter repos if specified
    repos_to_test = REPOS
    if args.repos:
        repos_to_test = [r for r in REPOS if r.name in args.repos]
        if not repos_to_test:
            error(f"No matching repos found. Available: {[r.name for r in REPOS]}")
            return 1
    
    # Show available strategies
    from src.expert import DeployStrategy
    available_strategies = [s.name.lower() for s in DeployStrategy]
    info(f"Available strategies: {available_strategies}")
    print()
    
    # Run tests
    results: List[TestResult] = []
    
    import time
    total_start = time.time()
    
    for i, repo in enumerate(repos_to_test, 1):
        info(f"[{i}/{len(repos_to_test)}] Testing {repo.name}...")
        
        result = run_repo(repo, strategy=args.strategy)
        results.append(result)
        
        print()
    
    total_elapsed = time.time() - total_start
    info(f"Total time: {total_elapsed:.1f}s")
    
    # Print summary
    print_summary(results)
    
    # Return status
    all_ok = all(r.deploy_ok and r.run_ok for r in results)
    if all_ok:
        success("ALL TESTS PASSED!")
        return 0
    else:
        error("SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
