#!/usr/bin/env python3
"""
ALE-Bench Runner

Runs the Kapso Agent on AtCoder Heuristic Contest problems from ALE-Bench.

Usage:
    python -m benchmarks.ale.runner --problem <problem_id>
    python -m benchmarks.ale.runner --problem ahc039
    python -m benchmarks.ale.runner --problem ahc039 --developer-agent gemini
    python -m benchmarks.ale.runner --list  # List available problems
    
Options:
    --problem, -p           Problem ID to solve (e.g., ahc039)
    --iterations, -i        Maximum iterations (default: 14)
    --mode, -m              Config mode: ALE_CONFIGS, HEAVY_THINKING, MINIMAL
    --coding-agent, -d      Coding agent: aider, gemini, claude_code, openhands
    --list                  List all available problems
    --lite                  List only lite benchmark problems
"""

import argparse
import os
import sys
import warnings
import yaml
from typing import Optional

# Suppress ResourceWarning from ale_bench library (unclosed image files)
warnings.filterwarnings("ignore", category=ResourceWarning, module="ale_bench")
warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed file")

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

import ale_bench

from kapso.execution.orchestrator import OrchestratorAgent
from kapso.execution.coding_agents.factory import CodingAgentFactory
from benchmarks.ale.handler import AleBench

# Path to ALE-Bench specific configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

# Available coding agents
AVAILABLE_AGENTS = ["aider", "gemini", "claude_code", "openhands"]


def list_problems(lite_only: bool = False) -> None:
    """List available ALE-Bench problems."""
    handler = AleBench()
    
    if lite_only:
        problems = handler.get_lite_problems_list()
        print(f"\n{'='*60}")
        print(f"ALE-Bench Lite Problems ({len(problems)} total)")
        print(f"{'='*60}")
    else:
        problems = ale_bench.list_problem_ids()
        print(f"\n{'='*60}")
        print(f"All ALE-Bench Problems ({len(problems)} total)")
        print(f"{'='*60}")
    
    for problem_id in sorted(problems):
        print(f"  • {problem_id}")
    print()


def list_agents() -> None:
    """List available coding agents with detailed info from agents.yaml."""
    CodingAgentFactory.print_agents_info()


def solve_problem(
    problem_id: str,
    max_iterations: int = 14,
    mode: Optional[str] = None,
    coding_agent: Optional[str] = None,
    use_kg: bool = False,
) -> dict:
    """
    Solve a single ALE-Bench problem.
    
    Args:
        problem_id: The AtCoder problem ID (e.g., ahc039)
        max_iterations: Maximum experiment iterations
        mode: Config mode (ALE_CONFIGS, HEAVY_THINKING, MINIMAL)
        coding_agent: Coding agent to use (aider, gemini, claude_code, openhands)
        use_kg: Whether to use the knowledge graph
        
    Returns:
        Dictionary with cost and evaluation results
    """
    print(f"\n{'='*60}")
    print(f"Solving: {problem_id}")
    print(f"{'='*60}")
    print(f"  Max iterations: {max_iterations}")
    print(f"  Config mode: {mode or 'default (ALE_CONFIGS)'}")
    print(f"  Coding agent: {coding_agent or 'from config'}")
    print(f"  Knowledge graph: {'enabled' if use_kg else 'disabled'}")
    print()
    
    # Initialize handler and orchestrator
    problem_handler = AleBench(problem_id)
    orchestrator = OrchestratorAgent(
        problem_handler,
        config_path=CONFIG_PATH,
        mode=mode,
        coding_agent=coding_agent,
        is_kg_active=use_kg,
    )
    
    # Run the solve loop
    orchestrator.solve(experiment_max_iter=max_iterations)

    # Checkout best solution and evaluate
    orchestrator.search_strategy.checkout_to_best_experiment_branch()
    cost = orchestrator.get_cumulative_cost()
    
    workspace = orchestrator.search_strategy.workspace.workspace_dir
    
    print(f"\nBest solution at: {workspace}")
    print(f"Total cost: ${cost:.3f}")
    
    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    private_evaluation = problem_handler.final_evaluate(workspace)
    
    result = {
        "problem_id": problem_id,
        "cost": f"${cost:.3f}",
        "solution_path": workspace,
        "private_evaluation": private_evaluation,
    }
    
    print(f"\nResults: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run Kapso Agent on ALE-Bench problems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "-p", "--problem",
        type=str,
        help="Problem ID to solve (e.g., ahc039)"
    )
    parser.add_argument(
        "-i", "--iterations",
        type=int,
        default=30,
        help="Maximum experiment iterations (default: 30)"
    )
    parser.add_argument(
        "-m", "--mode",
        type=str,
        default=None,
        help="Configuration mode (default: ALE_CONFIGS)"
    )
    parser.add_argument(
        "-d", "--coding-agent",
        type=str,
        choices=AVAILABLE_AGENTS,
        default=None,
        help="Coding agent to use: aider, gemini, claude_code, openhands"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available problems"
    )
    parser.add_argument(
        "--lite",
        action="store_true",
        help="List only lite benchmark problems"
    )
    parser.add_argument(
        "--list-agents",
        action="store_true",
        help="List available coding agents"
    )
    
    args = parser.parse_args()
    
    # Handle list commands
    if args.list_agents:
        list_agents()
        return
    
    if args.list or args.lite:
        list_problems(lite_only=args.lite)
        return
    
    # Require problem ID for solving
    if not args.problem:
        parser.print_help()
        print("\nError: --problem is required unless using --list or --lite")
        sys.exit(1)
    
    # Verify problem exists
    all_problems = ale_bench.list_problem_ids()
    if args.problem not in all_problems:
        print(f"\nError: Unknown problem '{args.problem}'")
        print("Use --list to see available problems")
        sys.exit(1)
    
    # Load config to get use_knowledge_graph
    try:
        with open(CONFIG_PATH, 'r') as f:
            config_data = yaml.safe_load(f)
        mode_name = args.mode or config_data.get('default_mode', 'ALE_CONFIGS')
        mode_config = config_data.get('modes', {}).get(mode_name, {})
        use_knowledge_graph = mode_config.get('use_knowledge_graph', False)
    except Exception as e:
        print(f"Warning: Could not load config for KG setting: {e}")
        use_knowledge_graph = False
    
    # Solve the problem
    result = solve_problem(
        problem_id=args.problem,
        max_iterations=args.iterations,
        mode=args.mode,
        coding_agent=args.coding_agent,
        use_kg=use_knowledge_graph,
    )
    
    print("\n" + "="*60)
    print("COMPLETED")
    print("="*60)
    print(f"Problem: {result['problem_id']}")
    print(f"Cost: {result['cost']}")
    print(f"Solution: {result['solution_path']}")


if __name__ == "__main__":
    main()
