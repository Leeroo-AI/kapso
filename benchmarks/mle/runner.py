#!/usr/bin/env python3
"""
MLE-Bench Runner

Runs the Kapso Agent on Kaggle ML competitions from MLE-Bench.

Usage:
    python -m benchmarks.mle.runner --competition <competition_id>
    python -m benchmarks.mle.runner --competition tabular-playground-series-dec-2021
    python -m benchmarks.mle.runner --competition tabular-playground-series-dec-2021 --developer-agent gemini
    python -m benchmarks.mle.runner --list  # List available competitions
    
Options:
    --competition, -c       Competition ID to solve
    --iterations, -i        Maximum iterations (default: 20)
    --mode, -m              Config mode: MLE_CONFIGS, HEAVY_EXPERIMENTATION, MINIMAL
    --coding-agent, -d      Coding agent: aider, gemini, claude_code, openhands
    --no-kg                 Disable knowledge graph
    --list                  List all available competitions
    --lite                  List only lite benchmark competitions
"""

import argparse
import os
import sys
from typing import Optional

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

import mlebench
from mlebench.registry import registry

from kapso.execution.orchestrator import OrchestratorAgent
from kapso.execution.coding_agents.factory import CodingAgentFactory
from benchmarks.mle.handler import MleBenchHandler

# Path to MLE-Bench specific configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

# Available coding agents
AVAILABLE_AGENTS = ["aider", "gemini", "claude_code", "openhands"]


def list_competitions(lite_only: bool = False) -> None:
    """List available MLE-Bench competitions."""
    if lite_only:
        competitions = registry.get_lite_competition_ids()
        print(f"\n{'='*60}")
        print(f"MLE-Bench Lite Competitions ({len(competitions)} total)")
        print(f"{'='*60}")
    else:
        competitions = registry.list_competition_ids()
        print(f"\n{'='*60}")
        print(f"All MLE-Bench Competitions ({len(competitions)} total)")
        print(f"{'='*60}")
    
    for comp_id in sorted(competitions):
        print(f"  • {comp_id}")
    print()


def list_agents() -> None:
    """List available coding agents with detailed info from agents.yaml."""
    CodingAgentFactory.print_agents_info()


def solve_competition(
    competition_id: str,
    max_iterations: int = 20,
    mode: Optional[str] = None,
    use_kg: bool = True,
    coding_agent: Optional[str] = None,
) -> dict:
    """
    Solve a single MLE-Bench competition.
    
    Args:
        competition_id: The Kaggle competition ID
        max_iterations: Maximum experiment iterations
        mode: Config mode (MLE_CONFIGS, HEAVY_EXPERIMENTATION, MINIMAL)
        use_kg: Whether to use the knowledge graph
        coding_agent: Coding agent to use (aider, gemini, claude_code, openhands)
        
    Returns:
        Dictionary with cost and evaluation results
    """
    # Load config to get settings
    import yaml
    with open(CONFIG_PATH, 'r') as f:
        config_data = yaml.safe_load(f)
    active_mode = mode or config_data.get('default_mode', 'MLE_CONFIGS')
    mode_config = config_data.get('modes', {}).get(active_mode, {})
    fetch_huggingface = mode_config.get('fetch_huggingface_models', True)
    
    # use_kg from CLI overrides config (--no-kg forces disable)
    # If CLI didn't disable it, check config
    use_kg_config = mode_config.get('use_knowledge_graph', True)
    effective_use_kg = use_kg and use_kg_config
    
    print(f"\n{'='*60}")
    print(f"Solving: {competition_id}")
    print(f"{'='*60}")
    print(f"  Max iterations: {max_iterations}")
    print(f"  Config mode: {active_mode}")
    print(f"  Coding agent: {coding_agent or 'from config'}")
    print(f"  Knowledge graph: {'enabled' if effective_use_kg else 'disabled'}")
    print(f"  Fetch HuggingFace models: {fetch_huggingface}")
    print()
    
    # Initialize handler and orchestrator
    problem_handler = MleBenchHandler(competition_id, fetch_huggingface_models=fetch_huggingface)
    orchestrator = OrchestratorAgent(
        problem_handler,
        is_kg_active=effective_use_kg,
        config_path=CONFIG_PATH,
        mode=mode,
        coding_agent=coding_agent,
    )
    
    # Run the solve loop
    orchestrator.solve(experiment_max_iter=max_iterations)
    
    # Get results
    print("\n" + "="*60)
    print("Experiment History:")
    print("="*60)
    print(orchestrator.search_strategy.get_experiment_history())
    
    # Checkout best solution and evaluate
    orchestrator.search_strategy.checkout_to_best_experiment_branch()
    cost = orchestrator.get_cumulative_cost()
    
    workspace = orchestrator.search_strategy.workspace.workspace_dir
    branch = orchestrator.search_strategy.workspace.get_current_branch()
    solution_path = f"{workspace}/{branch}"
    
    print(f"\nBest solution at: {solution_path}")
    print(f"Total cost: ${cost:.3f}")
    
    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation (Private Test Set)")
    print("="*60)
    private_evaluation = problem_handler.final_evaluate(solution_path)
    
    result = {
        "competition_id": competition_id,
        "cost": f"${cost:.3f}",
        "solution_path": solution_path,
        "private_evaluation": private_evaluation,
    }
    
    print(f"\nResults: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run Kapso Agent on MLE-Bench competitions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "-c", "--competition",
        type=str,
        help="Competition ID to solve (e.g., tabular-playground-series-dec-2021)"
    )
    parser.add_argument(
        "-i", "--iterations",
        type=int,
        default=20,
        help="Maximum experiment iterations (default: 20)"
    )
    parser.add_argument(
        "-m", "--mode",
        type=str,
        default=None,
        help="Configuration mode (default: MLE_CONFIGS)"
    )
    parser.add_argument(
        "-d", "--coding-agent",
        type=str,
        choices=AVAILABLE_AGENTS,
        default=None,
        help="Coding agent to use: aider, gemini, claude_code, openhands"
    )
    parser.add_argument(
        "--no-kg",
        action="store_true",
        help="Disable knowledge graph"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available competitions"
    )
    parser.add_argument(
        "--lite",
        action="store_true",
        help="List only lite benchmark competitions"
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
        list_competitions(lite_only=args.lite)
        return
    
    # Require competition ID for solving
    if not args.competition:
        parser.print_help()
        print("\nError: --competition is required unless using --list or --lite")
        sys.exit(1)
    
    # Verify competition exists
    all_competitions = registry.list_competition_ids()
    if args.competition not in all_competitions:
        print(f"\nError: Unknown competition '{args.competition}'")
        print("Use --list to see available competitions")
        sys.exit(1)
    
    # Solve the competition
    result = solve_competition(
        competition_id=args.competition,
        max_iterations=args.iterations,
        mode=args.mode,
        use_kg=not args.no_kg,
        coding_agent=args.coding_agent,
    )
    
    print("\n" + "="*60)
    print("COMPLETED")
    print("="*60)
    print(f"Competition: {result['competition_id']}")
    print(f"Cost: {result['cost']}")
    print(f"Solution: {result['solution_path']}")


if __name__ == "__main__":
    main()
