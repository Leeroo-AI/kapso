#!/usr/bin/env python3
# Kapso Agent CLI
#
# Command-line interface for the Kapso Agent system.
#
# Usage:
#     python -m src.cli --goal "Build a web scraper..."
#     python -m src.cli --goal-file problem.txt
#     python -m src.cli --goal "Build a classifier" --iterations 20
#
# Options:
#     --goal, -g              Goal/problem description (inline)
#     --goal-file, -f         File containing goal description
#     --iterations, -i        Maximum iterations (default: 10)
#     --mode, -m              Config mode: GENERIC, MINIMAL, TREE_SEARCH
#     --coding-agent, -a      Coding agent: aider, gemini, claude_code, openhands
#     --main-file             Entry point file (default: main.py)
#     --language              Programming language (default: python)
#     --timeout               Execution timeout in seconds (default: 300)
#     --output                Output directory for the solution
#     --eval-dir              Directory with evaluation files
#     --data-dir              Directory with data files
#     --initial-repo          Initial repository (local path or GitHub URL)

import argparse
import sys
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from src.kapso import Kapso, Source
from src.execution.coding_agents.factory import CodingAgentFactory


# Available coding agents
AVAILABLE_AGENTS = ["aider", "gemini", "claude_code", "openhands"]


def list_agents() -> None:
    """List available coding agents with detailed info."""
    CodingAgentFactory.print_agents_info()


def main():
    parser = argparse.ArgumentParser(
        description="Kapso Agent - Build robust software from goals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple usage
  python -m src.cli --goal "Build a web scraper for news articles"
  
  # With data and evaluation directories
  python -m src.cli --goal "Build a classifier" \\
      --eval-dir ./eval/ \\
      --data-dir ./data/
  
  # Full options
  python -m src.cli --goal-file problem.txt \\
      --iterations 20 \\
      --coding-agent claude_code \\
      --initial-repo https://github.com/owner/repo \\
      --output ./my_solution
"""
    )
    
    # Goal specification
    goal_group = parser.add_mutually_exclusive_group()
    goal_group.add_argument(
        "-g", "--goal",
        type=str,
        help="Goal/problem description (inline)"
    )
    goal_group.add_argument(
        "-f", "--goal-file",
        type=str,
        help="File containing goal description"
    )
    
    # Basic options
    parser.add_argument(
        "-i", "--iterations",
        type=int,
        default=10,
        help="Maximum experiment iterations (default: 10)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory for the solution"
    )
    
    # Configuration options
    parser.add_argument(
        "-m", "--mode",
        type=str,
        default=None,
        help="Configuration mode (GENERIC, MINIMAL, TREE_SEARCH)"
    )
    parser.add_argument(
        "-a", "--coding-agent",
        type=str,
        choices=AVAILABLE_AGENTS,
        default=None,
        help="Coding agent to use"
    )
    
    # Directory options (new design)
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=None,
        help="Directory with evaluation files (copied to kapso_evaluation/)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory with data files (copied to kapso_datasets/)"
    )
    parser.add_argument(
        "--initial-repo",
        type=str,
        default=None,
        help="Initial repository (local path or GitHub URL)"
    )
    
    # List commands
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
    
    # Get goal text
    if args.goal_file:
        with open(args.goal_file) as f:
            goal = f.read()
    elif args.goal:
        goal = args.goal
    else:
        parser.print_help()
        print("\nError: --goal or --goal-file required")
        sys.exit(1)
    
    # Create expert
    kapso = Kapso()
    
    # Build solution
    solution = kapso.evolve(
        goal=goal,
        output_path=args.output,
        max_iterations=args.iterations,
        mode=args.mode,
        coding_agent=args.coding_agent,
        eval_dir=args.eval_dir,
        data_dir=args.data_dir,
        initial_repo=args.initial_repo,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)
    print(f"Solution: {solution.code_path}")
    print(f"Goal achieved: {solution.succeeded}")
    if solution.final_score is not None:
        print(f"Final score: {solution.final_score}")
    print(f"Cost: {solution.metadata.get('cost', 'N/A')}")
    print(f"Stopped reason: {solution.metadata.get('stopped_reason', 'N/A')}")


if __name__ == "__main__":
    main()

