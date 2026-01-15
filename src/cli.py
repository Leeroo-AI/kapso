#!/usr/bin/env python3
# Tinkerer Agent CLI
#
# Command-line interface for the Tinkerer Agent system.
#
# Usage:
#     python -m src.cli --goal "Build a web scraper..."
#     python -m src.cli --goal-file problem.txt
#     python -m src.cli --goal "Build a classifier" --evaluator regex_pattern --iterations 20
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
#     --evaluator             Evaluator type: no_score, regex_pattern, llm_judge
#     --stop-condition        Stop condition: never, threshold, plateau
#     --output                Output directory for the solution

import argparse
import sys
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from src.tinkerer import Tinkerer, Source
from src.execution.coding_agents.factory import CodingAgentFactory
from src.environment.evaluators import EvaluatorFactory
from src.environment.stop_conditions import StopConditionFactory


# Available coding agents
AVAILABLE_AGENTS = ["aider", "gemini", "claude_code", "openhands"]


def list_agents() -> None:
    """List available coding agents with detailed info."""
    CodingAgentFactory.print_agents_info()


def list_evaluators() -> None:
    """List available evaluators."""
    EvaluatorFactory.print_evaluators_info()


def list_stop_conditions() -> None:
    """List available stop conditions."""
    StopConditionFactory.print_conditions_info()


def main():
    parser = argparse.ArgumentParser(
        description="Tinkerer Agent - Build robust software from goals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple usage
  python -m src.cli --goal "Build a web scraper for news articles"
  
  # With evaluation
  python -m src.cli --goal "Build a classifier" \\
      --evaluator regex_pattern \\
      --stop-condition threshold
  
  # Full options
  python -m src.cli --goal-file problem.txt \\
      --iterations 20 \\
      --coding-agent gemini \\
      --evaluator llm_judge \\
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
        default=1,
        help="Maximum experiment iterations (default: 1)"
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
    
    # Execution options
    parser.add_argument(
        "--main-file",
        type=str,
        default="main.py",
        help="Entry point file (default: main.py)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="python",
        help="Programming language (default: python)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Execution timeout in seconds (default: 300)"
    )
    
    # Evaluation options
    parser.add_argument(
        "--evaluator",
        type=str,
        default="no_score",
        help="Evaluator type (default: no_score). Use --list-evaluators for options"
    )
    parser.add_argument(
        "--stop-condition",
        type=str,
        default="never",
        help="Stop condition (default: never). Use --list-stop-conditions for options"
    )
    
    # List commands
    parser.add_argument(
        "--list-agents",
        action="store_true",
        help="List available coding agents"
    )
    parser.add_argument(
        "--list-evaluators",
        action="store_true",
        help="List available evaluators"
    )
    parser.add_argument(
        "--list-stop-conditions",
        action="store_true",
        help="List available stop conditions"
    )
    
    args = parser.parse_args()
    
    # Handle list commands
    if args.list_agents:
        list_agents()
        return
    
    if args.list_evaluators:
        list_evaluators()
        return
    
    if args.list_stop_conditions:
        list_stop_conditions()
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
    tinkerer = Tinkerer()
    
    # Build solution
    solution = tinkerer.evolve(
        goal=goal,
        output_path=args.output,
        max_iterations=args.iterations,
        mode=args.mode,
        coding_agent=args.coding_agent,
        language=args.language,
        main_file=args.main_file,
        timeout=args.timeout,
        evaluator=args.evaluator,
        stop_condition=args.stop_condition,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)
    print(f"Solution: {solution.code_path}")
    print(f"Cost: {solution.metadata.get('cost', 'N/A')}")
    
    # Optionally learn from the solution for future runs
    # tinkerer.learn(Source.Solution(solution), wiki_dir="data/wikis")


if __name__ == "__main__":
    main()

