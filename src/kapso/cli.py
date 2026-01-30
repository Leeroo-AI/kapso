#!/usr/bin/env python3
# Kapso Agent CLI
#
# Command-line interface for the Kapso Agent system.
#
# Commands:
#     evolve    - Build software from goals
#     research  - Web research for objectives
#     learn     - Learn from knowledge sources
#     deploy    - Deploy solutions
#     index_kg  - Index knowledge graph
#
# Usage:
#     kapso evolve --goal "Build a web scraper..."
#     kapso research --objective "How to optimize transformers?"
#     kapso learn --repo https://github.com/user/repo
#     kapso deploy --solution-path ./solution
#     kapso index_kg --wiki-dir ./data/wikis --save-to ./data/indexes/ml.index

import argparse
import sys
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from kapso.kapso import Kapso, Source, DeployStrategy
from kapso.execution.coding_agents.factory import CodingAgentFactory
from kapso.researcher import ResearchMode, ResearchDepth


# Available coding agents
AVAILABLE_AGENTS = ["aider", "gemini", "claude_code", "openhands"]

# Available deploy strategies
DEPLOY_STRATEGIES = ["auto", "local", "docker", "modal", "bentoml", "langgraph"]

# Research modes
RESEARCH_MODES = ["idea", "implementation", "study"]

# Research depths
RESEARCH_DEPTHS = ["light", "deep"]


def list_agents() -> None:
    """List available coding agents with detailed info."""
    CodingAgentFactory.print_agents_info()


def cmd_evolve(args) -> None:
    """Handle the evolve command - build software from goals."""
    # Get goal text
    if args.goal_file:
        with open(args.goal_file) as f:
            goal = f.read()
    elif args.goal:
        goal = args.goal
    else:
        print("Error: --goal or --goal-file required for evolve command")
        sys.exit(1)
    
    # Create Kapso instance with optional KG index
    kapso = Kapso(kg_index=args.kg_index)
    
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


def cmd_research(args) -> None:
    """Handle the research command - web research for objectives."""
    # Get objective text
    if args.objective_file:
        with open(args.objective_file) as f:
            objective = f.read()
    elif args.objective:
        objective = args.objective
    else:
        print("Error: --objective or --objective-file required for research command")
        sys.exit(1)
    
    # Parse mode(s)
    # Default to ["idea", "implementation"] if not specified
    modes = args.mode if args.mode else ["idea", "implementation"]
    
    # Validate modes
    valid_modes = {"idea", "implementation", "study"}
    for m in modes:
        if m not in valid_modes:
            print(f"Error: Invalid mode '{m}'. Must be one of: idea, implementation, study")
            sys.exit(1)
    
    # If single mode, pass as string; if multiple, pass as list
    mode_arg = modes[0] if len(modes) == 1 else modes
    
    # Create Kapso instance
    kapso = Kapso()
    
    # Run research
    findings = kapso.research(
        objective=objective,
        mode=mode_arg,
        depth=args.depth,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("RESEARCH COMPLETE")
    print("=" * 60)
    
    # Print ideas if available
    if hasattr(findings, 'ideas'):
        ideas = findings.ideas
        if ideas:
            print("\n--- Ideas ---")
            for idea in ideas[:5]:
                print(f"  - {idea.source}: {idea.content[:100]}...")
    
    # Print implementations if available
    if hasattr(findings, 'implementations'):
        impls = findings.implementations
        if impls:
            print("\n--- Implementations ---")
            for impl in impls[:5]:
                print(f"  - {impl.source}: {impl.content[:100]}...")
    
    # Print report if available
    if hasattr(findings, 'report') and findings.report:
        print("\n--- Research Report ---")
        print(findings.report.content[:500] + "..." if len(findings.report.content) > 500 else findings.report.content)
    
    # Save to file if requested
    if args.output:
        import json
        output_data = {
            "objective": objective,
            "mode": modes,
            "depth": args.depth,
        }
        if hasattr(findings, 'ideas') and findings.ideas:
            output_data["ideas"] = [{"source": i.source, "content": i.content} for i in findings.ideas]
        if hasattr(findings, 'implementations') and findings.implementations:
            output_data["implementations"] = [{"source": i.source, "content": i.content} for i in findings.implementations]
        if hasattr(findings, 'report') and findings.report:
            output_data["report"] = findings.report.content
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


def cmd_learn(args) -> None:
    """Handle the learn command - learn from knowledge sources."""
    sources = []
    
    # Collect sources from arguments
    if args.repo:
        for repo_url in args.repo:
            sources.append(Source.Repo(repo_url))
    
    if args.solution:
        for solution_path in args.solution:
            sources.append(Source.Solution(solution_path))
    
    if not sources:
        print("Error: At least one source required (--repo or --solution)")
        sys.exit(1)
    
    # Create Kapso instance with optional KG index
    kapso = Kapso(kg_index=args.kg_index)
    
    # Run learning pipeline
    result = kapso.learn(
        *sources,
        wiki_dir=args.wiki_dir,
        skip_merge=args.skip_merge,
        kg_index=args.kg_index,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("LEARN COMPLETE")
    print("=" * 60)
    print(f"Sources processed: {result.sources_processed}")
    print(f"Pages extracted: {result.total_pages_extracted}")
    print(f"Created: {result.created}")
    print(f"Edited: {result.edited}")
    if result.errors:
        print(f"Errors: {len(result.errors)}")
        for err in result.errors[:5]:  # Show first 5 errors
            print(f"  - {err}")


def cmd_deploy(args) -> None:
    """Handle the deploy command - deploy solutions."""
    from kapso.execution.solution import SolutionResult
    
    # Create a SolutionResult from the provided path
    solution = SolutionResult(
        goal=args.goal or "Deployed solution",
        code_path=args.solution_path,
        experiment_logs=[],
        final_feedback=None,
        metadata={},
    )
    
    # Parse strategy
    strategy_map = {
        "auto": DeployStrategy.AUTO,
        "local": DeployStrategy.LOCAL,
        "docker": DeployStrategy.DOCKER,
        "modal": DeployStrategy.MODAL,
        "bentoml": DeployStrategy.BENTOML,
        "langgraph": DeployStrategy.LANGGRAPH,
    }
    strategy = strategy_map.get(args.strategy.lower(), DeployStrategy.AUTO)
    
    # Parse env vars
    env_vars = {}
    if args.env:
        for env_str in args.env:
            if '=' in env_str:
                key, value = env_str.split('=', 1)
                env_vars[key] = value
    
    # Create Kapso instance
    kapso = Kapso()
    
    # Deploy
    software = kapso.deploy(
        solution=solution,
        strategy=strategy,
        env_vars=env_vars if env_vars else None,
        coding_agent=args.coding_agent,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("DEPLOY COMPLETE")
    print("=" * 60)
    print(f"Strategy: {strategy}")
    print(f"Code path: {args.solution_path}")
    print(f"Software ready: {software.is_healthy()}")
    
    # If interactive mode, keep running
    if args.interactive:
        print("\nSoftware deployed. Press Ctrl+C to stop.")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping software...")
            software.stop()
            print("Stopped.")


def cmd_index_kg(args) -> None:
    """Handle the index_kg command - index knowledge graph."""
    if not args.save_to:
        print("Error: --save-to required for index_kg command")
        sys.exit(1)
    
    if not args.wiki_dir and not args.data_path:
        print("Error: --wiki-dir or --data-path required for index_kg command")
        sys.exit(1)
    
    # Create Kapso instance
    kapso = Kapso()
    
    # Index knowledge graph
    index_path = kapso.index_kg(
        wiki_dir=args.wiki_dir,
        data_path=args.data_path,
        save_to=args.save_to,
        search_type=args.search_type,
        force=args.force,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("INDEX COMPLETE")
    print("=" * 60)
    print(f"Index saved to: {index_path}")


def main():
    # Main parser
    parser = argparse.ArgumentParser(
        description="Kapso Agent - Build robust software from goals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  evolve     Build software from goals using experimentation
  research   Web research for objectives
  learn      Learn from knowledge sources (repos, solutions)
  deploy     Deploy solutions as running software
  index_kg   Index knowledge graph from wiki or JSON data

Examples:
  # Evolve a solution
  kapso evolve --goal "Build a web scraper"
  
  # Research a topic
  kapso research --objective "How to optimize transformers?"
  
  # Learn from a repository
  kapso learn --repo https://github.com/user/repo
  
  # Deploy a solution
  kapso deploy --solution-path ./solution --strategy local
  
  # Index knowledge graph
  kapso index_kg --wiki-dir ./data/wikis --save-to ./data/indexes/ml.index
"""
    )
    
    # Global options
    parser.add_argument(
        "--list-agents",
        action="store_true",
        help="List available coding agents"
    )
    
    # Subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # =========================================================================
    # EVOLVE command
    # =========================================================================
    evolve_parser = subparsers.add_parser(
        "evolve",
        help="Build software from goals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  kapso evolve --goal "Build a web scraper for news articles"
  kapso evolve --goal-file problem.txt --iterations 20
  kapso evolve --goal "Build a classifier" --eval-dir ./eval/ --data-dir ./data/
"""
    )
    
    # Goal specification
    goal_group = evolve_parser.add_mutually_exclusive_group()
    goal_group.add_argument("-g", "--goal", type=str, help="Goal/problem description")
    goal_group.add_argument("-f", "--goal-file", type=str, help="File containing goal")
    
    # Basic options
    evolve_parser.add_argument("-i", "--iterations", type=int, default=10, help="Max iterations (default: 10)")
    evolve_parser.add_argument("-o", "--output", type=str, help="Output directory")
    
    # Configuration options
    evolve_parser.add_argument("-m", "--mode", type=str, help="Config mode (GENERIC, MINIMAL)")
    evolve_parser.add_argument("-a", "--coding-agent", type=str, choices=AVAILABLE_AGENTS, help="Coding agent")
    
    # Directory options
    evolve_parser.add_argument("--eval-dir", type=str, help="Evaluation files directory")
    evolve_parser.add_argument("--data-dir", type=str, help="Data files directory")
    evolve_parser.add_argument("--initial-repo", type=str, help="Initial repository (path or GitHub URL)")
    
    # Knowledge graph
    evolve_parser.add_argument("--kg-index", type=str, help="Path to KG index file")
    
    # =========================================================================
    # RESEARCH command
    # =========================================================================
    research_parser = subparsers.add_parser(
        "research",
        help="Web research for objectives",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  kapso research --objective "How to optimize transformers?"
  kapso research --objective "LLM fine-tuning" --mode idea
  kapso research --objective "RAG implementation" --mode idea --mode implementation
  kapso research --objective-file topic.txt --mode study --depth deep
"""
    )
    
    # Objective specification
    obj_group = research_parser.add_mutually_exclusive_group()
    obj_group.add_argument("--objective", type=str, help="Research objective")
    obj_group.add_argument("--objective-file", type=str, help="File containing objective")
    
    # Research options
    research_parser.add_argument("--mode", type=str, action="append", help="Research mode: idea, implementation, study (can specify multiple)")
    research_parser.add_argument("--depth", type=str, choices=RESEARCH_DEPTHS, default="deep", help="Research depth (default: deep)")
    research_parser.add_argument("-o", "--output", type=str, help="Output file for results (JSON)")
    
    # =========================================================================
    # LEARN command
    # =========================================================================
    learn_parser = subparsers.add_parser(
        "learn",
        help="Learn from knowledge sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  kapso learn --repo https://github.com/user/repo
  kapso learn --repo https://github.com/user/repo1 --repo https://github.com/user/repo2
  kapso learn --solution ./my_solution --wiki-dir ./data/wikis
"""
    )
    
    # Source options (can specify multiple)
    learn_parser.add_argument("--repo", type=str, action="append", help="Repository URL (can specify multiple)")
    learn_parser.add_argument("--solution", type=str, action="append", help="Solution path (can specify multiple)")
    
    # Learning options
    learn_parser.add_argument("--wiki-dir", type=str, default="data/wikis", help="Wiki directory (default: data/wikis)")
    learn_parser.add_argument("--skip-merge", action="store_true", help="Skip merging into KG backends")
    learn_parser.add_argument("--kg-index", type=str, help="Path to KG index file")
    
    # =========================================================================
    # DEPLOY command
    # =========================================================================
    deploy_parser = subparsers.add_parser(
        "deploy",
        help="Deploy solutions as running software",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  kapso deploy --solution-path ./solution
  kapso deploy --solution-path ./solution --strategy docker
  kapso deploy --solution-path ./solution --env API_KEY=xxx --interactive
"""
    )
    
    # Required options
    deploy_parser.add_argument("--solution-path", type=str, required=True, help="Path to solution code")
    
    # Deploy options
    deploy_parser.add_argument("--strategy", type=str, choices=DEPLOY_STRATEGIES, default="auto", help="Deploy strategy (default: auto)")
    deploy_parser.add_argument("--goal", type=str, help="Goal description for the solution")
    deploy_parser.add_argument("--env", type=str, action="append", help="Environment variable (KEY=VALUE, can specify multiple)")
    deploy_parser.add_argument("--coding-agent", type=str, choices=AVAILABLE_AGENTS, default="claude_code", help="Coding agent for adaptation")
    deploy_parser.add_argument("--interactive", action="store_true", help="Keep running after deploy")
    
    # =========================================================================
    # INDEX_KG command
    # =========================================================================
    index_parser = subparsers.add_parser(
        "index_kg",
        help="Index knowledge graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  kapso index_kg --wiki-dir ./data/wikis --save-to ./data/indexes/ml.index
  kapso index_kg --data-path ./data/kg_data.json --save-to ./data/indexes/kaggle.index
  kapso index_kg --wiki-dir ./data/wikis --save-to ./data/indexes/ml.index --force
"""
    )
    
    # Data source (mutually exclusive)
    data_group = index_parser.add_mutually_exclusive_group()
    data_group.add_argument("--wiki-dir", type=str, help="Wiki directory to index")
    data_group.add_argument("--data-path", type=str, help="JSON data file to index")
    
    # Index options
    index_parser.add_argument("--save-to", type=str, required=True, help="Path to save .index file")
    index_parser.add_argument("--search-type", type=str, help="Search backend type (kg_graph_search, kg_llm_navigation)")
    index_parser.add_argument("--force", action="store_true", help="Clear existing data before indexing")
    
    # =========================================================================
    # Parse and execute
    # =========================================================================
    args = parser.parse_args()
    
    # Handle global options
    if args.list_agents:
        list_agents()
        return
    
    # Route to command handler
    if args.command == "evolve":
        cmd_evolve(args)
    elif args.command == "research":
        cmd_research(args)
    elif args.command == "learn":
        cmd_learn(args)
    elif args.command == "deploy":
        cmd_deploy(args)
    elif args.command == "index_kg":
        cmd_index_kg(args)
    else:
        parser.print_help()
        print("\nError: Please specify a command (evolve, research, learn, deploy, index_kg)")
        sys.exit(1)


if __name__ == "__main__":
    main()
