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
from pathlib import Path

import yaml

from kapso.core.config import compose_runtime_config, load_config
from kapso.cross_run.canonical import canonical_json_bytes, tree_or_blob_digest
from kapso.cross_run.launch.contracts import LaunchTaskContextRequest
from kapso.kapso import Kapso

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
CANONICAL_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "src",
    "kapso",
    "config.yaml",
)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
AVAILABLE_AGENTS = ["codex"]


def build_runtime_config(
    runtime_root: str,
    image_authority_path: str | None = None,
) -> str:
    """Compose the benchmark workload with the canonical scope registry."""

    runtime = compose_runtime_config(
        load_config(CANONICAL_CONFIG_PATH),
        load_config(CONFIG_PATH),
    )
    if image_authority_path is not None:
        image_path = Path(image_authority_path).expanduser().resolve(strict=True)
        image = json.loads(image_path.read_text(encoding="utf-8"))
        runtime["cross_run"]["launch"]["coding_agent_image"] = image
    runtime_directory = os.path.join(runtime_root, ".kapso_runtime")
    os.makedirs(runtime_directory, exist_ok=True)
    runtime_path = os.path.join(runtime_directory, "config.yaml")
    with open(runtime_path, "w", encoding="utf-8") as runtime_file:
        yaml.safe_dump(runtime, runtime_file, sort_keys=False)
    return runtime_path


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
    print(f"\n{'=' * 70}\nSolving: {args.dataset} / {args.task}\n{'=' * 70}")
    print(f"  Config mode: {args.mode or 'from config'}")
    print(f"  Coding agent: {args.coding_agent or 'from config'}")
    run_root = Path(args.workspace).expanduser().resolve(strict=False)
    runtime_contract = _read_object(Path(args.runtime_contract))
    runtime_config_path = build_runtime_config(
        str(run_root.parent),
        args.image_authority,
    )
    starting_sources = None
    task_context = None
    budget = None
    if not args.resume:
        starting_sources = {
            "relbench_starter_kit": (
                Path(DATA_DIR, "starter_kit").resolve(strict=True),
                "task/starter_kit",
            ),
            "relbench_evaluation": (
                Path(DATA_DIR, "generic_eval").resolve(strict=True),
                "task/evaluation",
            ),
        }
        task_context = _task_context_request(
            dataset=args.dataset,
            task=args.task,
            dependency_runtime_contract=runtime_contract,
            starting_artifact_refs=tuple(sorted(starting_sources)),
        )
        budget = {"fidelity": "full", "hardware": "configured_runtime"}
    knowledge = ""
    if args.knowledge_file is not None:
        knowledge = (
            Path(args.knowledge_file)
            .expanduser()
            .resolve(strict=True)
            .read_text(encoding="utf-8")
        )
    goal = f"Improve predictive modeling for RelBench {args.dataset}/{args.task}."
    solution = Kapso(config_path=runtime_config_path).evolve(
        goal=goal,
        output_path=str(run_root),
        task_context_request=task_context,
        starting_artifact_sources=starting_sources,
        dependency_runtime_contract=(None if args.resume else runtime_contract),
        budget_fidelity_envelope=budget,
        config_path=runtime_config_path,
        mode=args.mode or "RELBENCH_GENERIC",
        coding_agent=args.coding_agent,
        objective_direction="maximize",
        additional_context=knowledge,
        resume=args.resume,
    )
    report = {
        "dataset": args.dataset,
        "task": args.task,
        "workspace": solution.code_path,
        **solution.metadata,
    }
    print(json.dumps(report, indent=2, default=str))
    return report


def _read_object(path: Path) -> dict:
    normalized = path.expanduser().resolve(strict=True)
    payload = json.loads(normalized.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{normalized} must contain one non-empty JSON object")
    return payload


def _task_context_request(
    *,
    dataset: str,
    task: str,
    dependency_runtime_contract: dict,
    starting_artifact_refs: tuple[str, ...],
) -> LaunchTaskContextRequest:
    return LaunchTaskContextRequest.mint(
        capability_tags=("predict", "relational_tabular"),
        input_contract_fingerprint=tree_or_blob_digest(
            canonical_json_bytes({"dataset": dataset})
        ),
        target_contract_fingerprint=tree_or_blob_digest(
            canonical_json_bytes({"task": task})
        ),
        starting_artifact_refs=starting_artifact_refs,
        method_fingerprint=tree_or_blob_digest(b"relbench predictive modeling"),
        toolchain_fingerprint=tree_or_blob_digest(b"relbench python"),
        dependency_runtime_fingerprint=tree_or_blob_digest(
            canonical_json_bytes(dependency_runtime_contract)
        ),
        budget_hardware_envelope={"hardware": "configured_runtime"},
        transfer_dimensions={
            "dataset_family": "relational_tabular",
            "runtime_family": "python_ml",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Kapso Agent on RelBench tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-s", "--dataset", type=str, help="Dataset name (e.g. rel-f1)")
    parser.add_argument(
        "-t", "--task", type=str, help="Task name (e.g. driver-position)"
    )
    parser.add_argument("-m", "--mode", type=str, default="RELBENCH_GENERIC")
    parser.add_argument(
        "-d", "--coding-agent", type=str, choices=AVAILABLE_AGENTS, default=None
    )
    parser.add_argument("--workspace", type=str)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--runtime-contract", type=str)
    parser.add_argument("--image-authority", type=str)
    parser.add_argument("--knowledge-file", type=str, default=None)
    parser.add_argument(
        "--list", action="store_true", help="List native RelBench tasks"
    )
    parser.add_argument("--list-agents", action="store_true")
    args = parser.parse_args()

    if args.list_agents:
        from kapso.execution.coding_agents.factory import CodingAgentFactory

        CodingAgentFactory.print_agents_info()
        return
    if args.list:
        list_tasks()
        return
    missing_run_arguments = tuple(
        name
        for name, value in (
            ("--workspace", args.workspace),
            ("--runtime-contract", args.runtime_contract),
            ("--image-authority", args.image_authority),
        )
        if value is None
    )
    if missing_run_arguments:
        parser.error(
            f"{', '.join(missing_run_arguments)} required unless using "
            "--list or --list-agents"
        )
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
    print(f"Workspace: {result['workspace']}")
    print(f"Launch: {result['launch_manifest_id']}")
    print(f"Expert: {result['expert_release_id']}")
    print(f"Knowledge: {result['knowledge_snapshot_id']}")


if __name__ == "__main__":
    main()
