#!/usr/bin/env python3
"""
RelBench Runner

Runs the Kapso Agent on RelBench tasks (relbench.stanford.edu — relational deep
learning benchmark: 11 databases, 66 tasks across entity classification,
entity regression, recommendation, and autocomplete).

Usage:
    python -m benchmarks.relbench.runner --dataset rel-f1 --task driver-position
    python -m benchmarks.relbench.runner -s rel-hm -t user-item-purchase -i 25
    python -m benchmarks.relbench.runner --list            # list native tasks
    python -m benchmarks.relbench.runner --list-agents

Options:
    --dataset, -s        Dataset name (e.g. rel-f1)
    --task, -t           Task name (e.g. driver-position)
    --iterations, -i     Maximum search iterations (default: 20)
    --mode, -m           Config mode: RELBENCH_CONFIGS, HEAVY_EXPERIMENTATION, MINIMAL
    --coding-agent, -d   Coding agent: aider, gemini, claude_code, openhands
    --no-kg              Disable knowledge graph
    --workspace          Reuse/name a workspace dir (enables resuming archives)
    --resume             Resume from a checkpointed workspace
    --target-val         Stop early once validation primary metric reaches this value
    --rebuild-cache      Force rebuild of the sanitized data cache
    --knowledge-file     Extra knowledge markdown injected into the problem context
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
AVAILABLE_AGENTS = ["aider", "gemini", "claude_code", "openhands"]


def list_tasks() -> None:
    from relbench.tasks import get_task_names

    from benchmarks.relbench.task_specs import NATIVE_DATASETS

    print(f"\n{'=' * 70}\nRelBench native tasks (11 databases / 66 tasks)\n{'=' * 70}")
    for ds in NATIVE_DATASETS:
        names = get_task_names(ds)
        print(f"\n  {ds}:")
        for name in names:
            print(f"    • {name}")
    print()


def solve_task(args) -> dict:
    from kapso.execution.orchestrator import OrchestratorAgent

    from benchmarks.relbench.handler import RelBenchHandler

    print(f"\n{'=' * 70}\nSolving: {args.dataset} / {args.task}\n{'=' * 70}")
    print(f"  Max iterations: {args.iterations}")
    print(f"  Config mode: {args.mode or 'from config'}")
    print(f"  Coding agent: {args.coding_agent or 'from config'}")
    print(f"  Knowledge graph: {'disabled' if args.no_kg else 'enabled'}\n")

    handler = RelBenchHandler(
        dataset_name=args.dataset,
        task_name=args.task,
        planned_iterations=args.iterations,
        target_val_score=args.target_val,
        extra_knowledge_file=args.knowledge_file,
        rebuild_sanitized_cache=args.rebuild_cache,
    )

    # Strategy selection:
    # - tree (default): handler-scored benchmark_tree_search. No eval_dir —
    #   the handler computes official metrics itself, and passing a provided
    #   suite would turn on the integrity check against agents writing their
    #   own scripts into kapso_evaluation/.
    # - generic: champion-chain search with code-reading agentic ideation.
    #   Our provided grader (data/generic_eval) becomes the maintainer-
    #   registered evaluation entrypoint; in-loop scoring is val-only by
    #   construction (the sanitized cache holds no test labels).
    generic = args.strategy == "generic"
    mode = args.mode or ("RELBENCH_GENERIC" if generic else None)
    initial_repo = args.initial_repo
    if generic and not initial_repo:
        # The maintainer calibrates the registered evaluation at setup, which
        # requires a runnable candidate — and parent_policy=best needs a real
        # starting parent. Seed a trivial shape-correct baseline.
        import shutil
        import tempfile

        initial_repo = tempfile.mkdtemp(prefix="relbench_baseline_")
        shutil.copy2(
            os.path.join(DATA_DIR, "generic_baseline", "main.py"),
            os.path.join(initial_repo, "main.py"),
        )
    orchestrator = OrchestratorAgent(
        handler,
        config_path=CONFIG_PATH,
        mode=mode,
        coding_agent=args.coding_agent,
        is_kg_active=not args.no_kg,
        workspace_dir=args.workspace,
        resume=args.resume,
        initial_repo=initial_repo,
        eval_dir=os.path.join(DATA_DIR, "generic_eval") if generic else None,
        data_dir=os.path.join(DATA_DIR, "starter_kit"),
        goal=f"Beat the published state of the art on RelBench {args.dataset}/{args.task}",
    )

    orchestrator.solve(experiment_max_iter=args.iterations)

    print("\n" + "=" * 70 + "\nExperiment History\n" + "=" * 70)
    for node in orchestrator.search_strategy.get_experiment_history(best_last=True):
        print(f"  branch={node.branch_name} score={node.score} error={node.had_error}")

    best_branch = orchestrator.search_strategy.checkout_to_best_experiment_branch()
    cost = orchestrator.get_cumulative_cost()
    workspace = orchestrator.search_strategy.workspace.workspace_dir

    print("\n" + "=" * 70 + "\nFinal Evaluation (validation-selected, test reported once)\n" + "=" * 70)
    report = handler.final_evaluate(workspace)
    report.update({"best_branch": best_branch, "workspace": workspace, "cost_usd": round(cost, 3)})
    print(json.dumps(report, indent=2, default=str))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Kapso Agent on RelBench tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-s", "--dataset", type=str, help="Dataset name (e.g. rel-f1)")
    parser.add_argument("-t", "--task", type=str, help="Task name (e.g. driver-position)")
    parser.add_argument("-i", "--iterations", type=int, default=20)
    parser.add_argument("-m", "--mode", type=str, default=None)
    parser.add_argument("-d", "--coding-agent", type=str, choices=AVAILABLE_AGENTS, default=None)
    parser.add_argument("--no-kg", action="store_true")
    parser.add_argument("--workspace", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--strategy", type=str, choices=["tree", "generic"], default="tree",
        help="tree = handler-scored benchmark_tree_search (default); "
        "generic = champion-chain search with the provided grader",
    )
    parser.add_argument(
        "--initial-repo", type=str, default=None,
        help="Seed the workspace from an existing repo (e.g. a scout's winning branch)",
    )
    parser.add_argument("--target-val", type=float, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--knowledge-file", type=str, default=None)
    parser.add_argument("--list", action="store_true", help="List native RelBench tasks")
    parser.add_argument("--list-agents", action="store_true")
    args = parser.parse_args()

    if args.list_agents:
        from kapso.execution.coding_agents.factory import CodingAgentFactory

        CodingAgentFactory.print_agents_info()
        return
    if args.list:
        list_tasks()
        return
    if not args.dataset or not args.task:
        parser.print_help()
        print("\nError: --dataset and --task are required unless using --list")
        sys.exit(1)

    from relbench.tasks import get_task_names

    if args.task not in get_task_names(args.dataset):
        print(f"\nError: unknown task '{args.task}' for {args.dataset}.")
        print(f"Available: {get_task_names(args.dataset)}")
        sys.exit(1)

    result = solve_task(args)
    print("\n" + "=" * 70 + "\nCOMPLETED\n" + "=" * 70)
    print(f"Task: {args.dataset}/{args.task}")
    print(f"Selected run: {result.get('run')} | val: {result.get('val_metrics')}")
    print(f"TEST metrics: {result.get('test_metrics')}")
    print(f"Cost: ${result.get('cost_usd')}")


if __name__ == "__main__":
    main()
