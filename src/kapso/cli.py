#!/usr/bin/env python3
# Kapso Agent CLI
#
# Command-line interface for the Kapso Agent system.
#
# Commands:
#     evolve    - Build software from goals
#     research  - Web research for objectives
#     learn     - Trajectory learning (store, mining, updating, grading)
#     deploy    - Deploy solutions
#     index_kg  - Index knowledge graph
#
# Usage:
#     kapso evolve --goal "Build a web scraper..."
#     kapso research --objective "How to optimize transformers?"
#     kapso learn import --subset docs/plans/learning/d1-subset.yaml
#     kapso deploy --solution-path ./solution
#     kapso index_kg --wiki-dir ./data/wikis --save-to ./data/indexes/ml.index

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from dotenv import find_dotenv, load_dotenv
# Resolve .env from the caller's CWD upward. The no-arg default anchors at
# THIS file's directory — site-packages in a pip install — so the .env the
# README tells users to create next to their project was never read.
load_dotenv(find_dotenv(usecwd=True))

from kapso.kapso import Kapso, DeployStrategy, DEFAULT_CONFIG_PATH
from kapso.core.config import load_config, load_mode_config
from kapso.execution.inbox import list_registered_campaigns, render_requests
from kapso.execution.run_checkpoint import RunCheckpointStore
from kapso.core.preflight import (
    VERBS,
    configured_bank_home,
    dedupe,
    live_model_requirements,
    requirements_for,
)
from kapso.execution.coding_agents.factory import CodingAgentFactory
from kapso.learning.corpus_import import import_archive, import_subset
from kapso.learning.behavior import BehaviorRunner
from kapso.learning.codify_run import CodifyRunDriver
from kapso.learning.develop import DevelopmentDriver
from kapso.learning.graders.frame import GradingFrame
from kapso.learning.graders.gauntlet import GauntletRunner
from kapso.learning.graders.split import assert_batch_disjoint, load_split, validate_split
from kapso.learning.bank_remote import connect_bank, create_bank_repo
from kapso.learning.update_frame import UpdateFrame, init_bank
from kapso.learning.mining import MiningFrame
from kapso.learning.trajectory_store import TrajectoryStore


# Available coding agents
AVAILABLE_AGENTS = CodingAgentFactory.list_available()

# Available deploy strategies
DEPLOY_STRATEGIES = ["auto", "local", "docker", "modal", "bentoml", "langgraph"]

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
        time_budget_minutes=args.time_budget_minutes,
        cost_budget=args.cost_budget,
        finalization_reserve_minutes=args.finalization_reserve_minutes,
        mode=args.mode,
        coding_agent=args.coding_agent,
        eval_dir=args.eval_dir,
        data_dir=args.data_dir,
        initial_repo=args.initial_repo,
        resume=args.resume,
    )
    
    _print_evolve_summary(solution)


def _print_evolve_summary(solution) -> None:
    """The block after a campaign: COMPLETED, or WAITING ON YOU when it
    paused for a reply in the inbox (design v4 §4.6). A pause is not a
    failure: the exit code stays 0."""
    print("\n" + "=" * 60)
    if solution.metadata.get("stopped_reason") == "waiting_for_user":
        campaign = str(Path(solution.code_path).resolve())
        requests = solution.requests
        target = f" {requests[0]['id']}" if len(requests) > 1 else ""
        print("WAITING ON YOU")
        print("=" * 60)
        print(f"Solution so far: {campaign}")
        if solution.final_score is not None:
            print(f"Best score so far: {solution.final_score}")
        print(f'Reply with:  kapso inbox reply{target} "…"   (inside {campaign})')
        print(f"Any time:    kapso inbox {campaign}")
        return
    print("COMPLETED")
    print("=" * 60)
    print(f"Solution: {solution.code_path}")
    print(f"Goal achieved: {solution.succeeded}")
    if solution.final_score is not None:
        print(f"Final score: {solution.final_score}")
    print(f"Cost: {solution.metadata.get('cost', 'N/A')}")
    print(f"Stopped reason: {solution.metadata.get('stopped_reason', 'N/A')}")


# =============================================================================
# INBOX (docs/research/evolve-hub-design.md v4 §4.6)
# =============================================================================

# A reply that looks like a credential. Replies are stored in plain text
# and handed to the coder as text, so the value belongs in the file the
# fix names, never here. Shapes of common tokens — structural, not knobs.
_SECRET_SHAPES = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"hf_[A-Za-z0-9]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"xox[abp]-[A-Za-z0-9-]{20,}"),
)


def _looks_like_a_secret(note: str) -> bool:
    return any(shape.search(note) for shape in _SECRET_SHAPES)


def _campaign_here(start: Path) -> Optional[Path]:
    """The nearest ancestor of `start` that is a campaign directory."""
    for candidate in (start, *start.parents):
        if (candidate / ".kapso" / "run_state.json").is_file():
            return candidate
    return None


def _inbox_campaign(explicit: Optional[str]) -> Optional[Path]:
    """The campaign a `kapso inbox` invocation means: the named directory,
    else the campaign the cwd is inside, else None."""
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"{explicit} is not a directory")
        return path.resolve()
    return _campaign_here(Path.cwd().resolve())


def _campaign_goal(campaign: Path) -> str:
    store = RunCheckpointStore(str(campaign))
    if not store.exists():
        return ""
    goal = store.load().goal.strip()
    return goal.splitlines()[0] if goal else ""


def _waiting_since(requests) -> str:
    newest = max(request.requested_at for request in requests)
    seconds = (datetime.now(timezone.utc) - datetime.fromisoformat(newest)).total_seconds()
    if seconds < 3600:
        return f"{max(1, int(seconds // 60))}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"


def _render_campaign_inbox(campaign: Path) -> str:
    requests = Kapso.inbox(str(campaign))
    if not requests:
        return f"Nothing waiting on you in {campaign}."
    label = f"  waiting {_waiting_since(requests)}"
    if len(requests) > 1:
        label += f" · {len(requests)} requests"
    goal = _campaign_goal(campaign)
    header = f"{campaign}  {goal}{label}" if goal else f"{campaign}{label}"
    target = f" {requests[0].id}" if len(requests) > 1 else ""
    return "\n".join([
        header,
        "",
        render_requests(requests, Kapso.inbox_ideas(str(campaign))),
        "",
        f'  reply with   kapso inbox reply{target} "…"',
    ])


def cmd_inbox(args) -> None:
    """`kapso inbox [CAMPAIGN]` — what is waiting on you; `kapso inbox
    reply [CAMPAIGN] [ID] [NOTE]` — answer it, and the campaign resumes."""
    words = list(args.words)
    if words and words[0] == "reply":
        cmd_inbox_reply(words[1:], yes=args.yes)
        return
    if len(words) > 1:
        print("Error: kapso inbox takes at most a campaign directory")
        sys.exit(1)
    campaign = _inbox_campaign(words[0] if words else None)
    if campaign is not None:
        print(_render_campaign_inbox(campaign))
        return
    registry = load_mode_config(args.config or DEFAULT_CONFIG_PATH)["inbox"]["registry"]
    waiting = [
        Path(entry["path"])
        for entry in list_registered_campaigns(registry)
        if Kapso.inbox(entry["path"])
    ]
    if not waiting:
        print("Nothing waiting on you.")
        return
    print("\n\n".join(_render_campaign_inbox(campaign) for campaign in waiting))


def cmd_inbox_reply(words, *, yes: bool) -> None:
    """Answer one request; the campaign resumes here when it can. The id
    may be left out when one request is open; an empty note means done."""
    words = list(words)
    explicit = None
    if words and not words[0].isdigit() and Path(words[0]).expanduser().is_dir():
        explicit = words.pop(0)
    campaign = _inbox_campaign(explicit)
    if campaign is None:
        print("Error: not inside a campaign — name it: kapso inbox reply <campaign> [id] [note]")
        sys.exit(1)
    if words and words[0].isdigit():
        request_id = int(words.pop(0))
    else:
        open_requests = Kapso.inbox(str(campaign))
        if len(open_requests) != 1:
            print("Error: which request? kapso inbox reply <id> [note]")
            sys.exit(1)
        request_id = open_requests[0].id
    note = " ".join(words)
    if _looks_like_a_secret(note) and not yes:
        print(
            "This reply looks like a credential. Replies are stored in plain "
            "text and handed to the coder as text; put the value where the "
            "fix says (usually the .env the run loaded) and reply with a note "
            "instead. To store it anyway, add --yes at the end."
        )
        sys.exit(1)
    solution = Kapso.reply(str(campaign), request_id, note)
    if solution is not None:
        _print_evolve_summary(solution)


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
    """Handle the learn command group — the learn-from-trajectories system."""
    config = load_config(args.config or DEFAULT_CONFIG_PATH)
    if args.learn_command == "import":
        store = TrajectoryStore.from_config(config)
        upload = False if args.no_upload else None
        if bool(args.subset) == bool(args.archive):
            print("Error: exactly one of --subset or --archive is required")
            sys.exit(1)
        if args.subset:
            report_path = import_subset(
                store, args.subset, config["learning"]["import_report_dir"], upload
            )
            print(f"Import report: {report_path}")
        else:
            outcome = import_archive(store, args.archive, args.id, upload)
            print(f"{outcome['id']}: {outcome['status']}")
    elif args.learn_command == "mine":
        frame = MiningFrame.from_config(config)
        if bool(args.trajectory) == bool(args.all):
            print("Error: exactly one of --trajectory or --all is required")
            sys.exit(1)
        targets = (
            [args.trajectory] if args.trajectory
            else sorted(
                m["id"] for m in frame.store.list_manifests()
                if not m.get("derived", {}).get("mined")
            )
        )
        for trajectory_id in targets:
            mined_dir = frame.mine(trajectory_id, force=args.force)
            print(f"{trajectory_id}: mined -> {mined_dir}")
    elif args.learn_command == "grade":
        grading = GradingFrame(TrajectoryStore.from_config(config), config)
        run_root = config["learning"]["graders"]["run_root"]
        if bool(args.split) == bool(args.trajectory):
            print("Error: exactly one of --split (full) or --trajectory (exam) is required")
            sys.exit(1)
        if args.split:
            if not args.learner_version:
                print("Error: --learner-version is required in full mode")
                sys.exit(1)
            split = load_split(args.split)
            findings = validate_split(split, grading.store.list_manifests())
            if findings:
                print("Split validation failed:")
                for finding in findings:
                    print(f"  - {finding}")
                sys.exit(1)
            run_dir = grading.grade_full(
                split, args.bank, args.bank_head, args.learner_version, run_root
            )
            print(f"Scorecard: {run_dir / 'scorecard.yaml'}")
        else:
            # Operating regime: the local store IS the bank's past, so the
            # allowed surface is every other mined trajectory. Development
            # replays never come through here — the driver passes the
            # ingested-so-far surface itself.
            learn_set_ids = [
                manifest["id"]
                for manifest in grading.store.list_manifests()
                if manifest["id"] != args.trajectory
                and (grading.store.local / manifest["id"] / "mined").is_dir()
            ]
            result = grading.grade_exam(
                args.trajectory, args.bank, args.bank_head, run_root,
                learn_set_ids,
            )
            print(f"Exam report: {result}")
    elif args.learn_command == "update":
        store = TrajectoryStore.from_config(config)
        batch = []
        if args.batch_manifest:
            with open(args.batch_manifest) as handle:
                batch = yaml.safe_load(handle) or []
        if args.split:
            split = load_split(args.split)
            assert_batch_disjoint(split, [item["trajectory"] for item in batch])
        frame = UpdateFrame(store, config)
        run_dir = frame.run_update(
            batch, config["learning"]["update_crew"]["run_root"], args.learner_version
        )
        print(f"Learner report: {run_dir / 'report.md'}")
    elif args.learn_command == "init-bank":
        init_bank(config["learning"]["bank"]["local_path"])
        print(f"Bank home created: {config['learning']['bank']['local_path']}")
    elif args.learn_command == "codify":
        # One codify run (CD§2) from a specialist's request; the verdict
        # lands beside the run for the next update transaction to fold (the
        # flip only ever commits with a green verdict in-transaction).
        store = TrajectoryStore.from_config(config)
        request = yaml.safe_load(Path(args.request).read_text())
        bank_home = Path(config["learning"]["bank"]["local_path"]).expanduser()
        card_name = request["card"]
        card_text = subprocess.run(
            ["git", "--git-dir", str(bank_home), "show",
             f"main:procedures/{card_name}/card.md"],
            check=True, capture_output=True, text=True,
        ).stdout
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        run_dir = (
            Path(config["learning"]["update_crew"]["run_root"]).expanduser()
            / "codify" / f"{card_name}-{stamp}"
        )
        driver = CodifyRunDriver(store, config)
        verdict = driver.run(request, card_text, str(run_dir))
        print(f"Codify run: {run_dir / 'verdict.yaml'}")
        print(f"Status: {verdict['status']} after {verdict['iterations']} iteration(s)")
    elif args.learn_command == "ingest":
        # Operating-regime chain (design §4.1 step 3): exam-before-lesson on
        # one arriving campaign — mine if needed, exam against the
        # production bank head, then the lesson (a one-trajectory update
        # run). The local store IS the past here; development replays use
        # the driver, never this path.
        store = TrajectoryStore.from_config(config)
        trajectory_id = args.trajectory
        manifest = store.manifest(trajectory_id)
        if not manifest.get("derived", {}).get("mined"):
            mined_dir = MiningFrame.from_config(config).mine(trajectory_id)
            print(f"Mined view: {mined_dir}")
        bank_home = Path(config["learning"]["bank"]["local_path"]).expanduser()
        graders_root = Path(config["learning"]["graders"]["run_root"]).expanduser()
        checkout = (
            graders_root / "ingest-serving"
            / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        )
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--quiet", str(bank_home), str(checkout)],
            check=True,
        )
        bank_head = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        grading = GradingFrame(store, config)
        learn_set_ids = [
            manifest_row["id"]
            for manifest_row in store.list_manifests()
            if manifest_row["id"] != trajectory_id
            and (store.local / manifest_row["id"] / "mined").is_dir()
        ]
        report_path = grading.grade_exam(
            trajectory_id, str(checkout), bank_head,
            str(graders_root), learn_set_ids,
        )
        print(f"Exam report: {report_path}")
        frame = UpdateFrame(store, config)
        run_dir = frame.run_update(
            [{"trajectory": trajectory_id, "hindcast_report": str(report_path)}],
            config["learning"]["update_crew"]["run_root"],
            args.learner_version,
        )
        print(f"Learner report: {run_dir / 'report.md'}")
    elif args.learn_command == "gauntlet":
        runner = GauntletRunner(TrajectoryStore.from_config(config), config)
        verdict = runner.run(args.learner_version)
        root = Path(config["learning"]["develop"]["run_root"]).expanduser()
        print(f"Gauntlet: {root / args.learner_version / 'gauntlet.md'}")
        print(f"Rolled verdict: {verdict}")
    elif args.learn_command == "develop":
        store = TrajectoryStore.from_config(config)
        split = load_split(args.split)
        findings = validate_split(split, store.list_manifests())
        if findings:
            print("Split validation failed:")
            for finding in findings:
                print(f"  - {finding}")
            sys.exit(1)
        driver = DevelopmentDriver(store, config)
        scorecard_dir = driver.run(split, args.learner_version)
        print(f"Scorecard: {scorecard_dir / 'scorecard.yaml'}")
    elif args.learn_command == "behave":
        runner = BehaviorRunner(TrajectoryStore.from_config(config), config)
        run_root = config["learning"]["behavior"]["run_root"]
        if bool(args.scenario) == bool(args.all):
            print("Error: exactly one of --scenario or --all is required")
            sys.exit(1)
        if args.scenario:
            result = runner.run_scenario(args.scenario, run_root)
            print(f"{result['scenario']}: {result['verdict']} — {result['rationale']}")
        else:
            rollup = runner.run_all(run_root)
            print(f"Behavior suite: {rollup['verdict']}")
            for row in rollup["scenarios"]:
                print(f"  {row['scenario']}: {row['verdict']}")


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


def cmd_bank(args) -> None:
    """Attach (and optionally create) the bank's share remote. Credentials
    are git's/gh's own business — kapso never handles them."""
    config = load_config(args.config or DEFAULT_CONFIG_PATH)
    home = Path(config["learning"]["bank"]["local_path"]).expanduser()

    if args.bank_command == "create":
        if shutil.which("gh") is None:
            print("`gh` CLI not found — create the repository yourself, then:")
            print(f"  kapso bank connect https://github.com/{args.slug}.git")
            sys.exit(1)
        url = create_bank_repo(args.slug, private=not args.public)
        print(f"created {url} ({'public' if args.public else 'private'})")
    else:
        url = args.url

    if not home.exists():
        init_bank(str(home))
        print(f"initialized bank home at {home}")
    connect_bank(home, url)
    print(f"bank connected: {home} -> {url}")
    print("lessons now push automatically at the end of every learn()")


def cmd_doctor(args) -> None:
    """Report what the active config actually needs, and exit 1 when a
    required item is missing. With no verb this is the union across all
    five verbs; naming one narrows it to that verb's own requirements.
    --models adds the live tier (a one-token probe per {cli, model} pair).

    Same resolver the verbs themselves run at call time, so the doctor can
    never disagree with what a run will do (core/preflight.py)."""
    config = load_config(args.config or DEFAULT_CONFIG_PATH)
    verb = args.verb
    # The only context a doctor can supply is what the config declares —
    # a live call's own arguments (skip_merge, strategy) are not knowable
    # here, so those verbs report their fullest requirement set.
    context = (
        {"bank_home": configured_bank_home(config)}
        if verb in ("evolve", "learn") else {}
    )

    requirements = requirements_for(verb, config, **context)
    if args.models:
        requirements = dedupe(
            requirements + live_model_requirements(verb, config, **context)
        )

    scope = f"kapso doctor {verb}" if verb else "kapso doctor"
    print(scope + "\n" + "=" * 68)
    for item in requirements:
        mark = "OK " if item.ok else ("FAIL" if item.required else "-- ")
        print(f"  [{mark}] {item.label}")
        if item.ok:
            continue
        if item.detail:
            print(f"         reason     {item.detail}")
        print(f"         needed by  {item.origin}")
        if item.fix:
            print(f"         fix        {item.fix}")

    failures = [i for i in requirements if not i.ok and i.required]
    advisories = [i for i in requirements if not i.ok and not i.required]
    print("=" * 68)
    if failures:
        print(f"{len(failures)} required check(s) failed — fix the items "
              f"above, then rerun `{scope}`.")
        sys.exit(1)
    summary = f"Required checks passed ({len(requirements)} checked)."
    if advisories:
        summary += (f" {len(advisories)} optional item(s) marked [-- ] "
                    "only limit features you may not need.")
    print(summary)


def cmd_watch(args) -> None:
    """One watch command for evolve / learn / learn_knowledge status files
    (observability design §3). Pure reader — never writes."""
    view = Kapso.status(args.path)
    if args.json:
        print(json.dumps(view.data, indent=1))
        return
    if not args.follow:
        print(view.explain())
        return
    # Follow: re-render at the operation's own heartbeat cadence (falls
    # back to a display-structural 5s when the file records none).
    interval = float(view.data.get("heartbeat_seconds") or 5)
    while True:
        view = Kapso.status(args.path)
        print("\x1b[2J\x1b[H" + view.explain(), flush=True)
        if view.state in ("done", "failed"):
            return
        time.sleep(interval)


def main():
    CodingAgentFactory.register_quiet_exit()
    # Main parser
    parser = argparse.ArgumentParser(
        description="Kapso Agent - Build robust software from goals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  evolve     Build software from goals using experimentation
  research   Web research for objectives
  learn      Trajectory learning (store, mining, updating, grading)
  deploy     Deploy solutions as running software
  index_kg   Index knowledge graph from wiki or JSON data
  watch      Watch a running evolve / learn / learn_knowledge operation
  inbox      What evolve is waiting on you for, and how to answer it
  doctor     Check this machine is ready to run Kapso

Examples:
  # Evolve a solution
  kapso evolve --goal "Build a web scraper"
  
  # Research a topic
  kapso research --objective "How to optimize transformers?"
  
  # Learn from a repository
  kapso learn import --subset docs/plans/learning/d1-subset.yaml
  
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
  kapso evolve --goal "Build a classifier" --output ./campaign --resume
"""
    )
    
    # Goal specification
    goal_group = evolve_parser.add_mutually_exclusive_group()
    goal_group.add_argument("-g", "--goal", type=str, help="Goal/problem description")
    goal_group.add_argument("-f", "--goal-file", type=str, help="File containing goal")
    
    # Basic options
    evolve_parser.add_argument("-i", "--iterations", type=int, default=10, help="Max iterations (default: 10)")
    evolve_parser.add_argument("-o", "--output", type=str, help="Output directory")
    evolve_parser.add_argument(
        "--time-budget-minutes",
        type=float,
        default=None,
        help="Wall-clock budget for the campaign (durable across resumes)",
    )
    evolve_parser.add_argument(
        "--cost-budget",
        type=float,
        default=None,
        help="Best-effort spend budget in USD",
    )
    evolve_parser.add_argument(
        "--finalization-reserve-minutes",
        type=float,
        default=None,
        help="Wall-clock escrowed for final checkout and evaluation",
    )
    evolve_parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue the compatible checkpoint in --output",
    )
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
    # LEARN command group — the learn-from-trajectories system
    # (docs/research/learn-from-trajectories-design.md; supersedes the old
    # wiki-source learner CLI per design §0 / Rule 7)
    # =========================================================================
    learn_parser = subparsers.add_parser(
        "learn",
        help="Trajectory learning: store, mining, updating, grading, serving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  kapso learn import --subset docs/plans/learning/d1-subset.yaml
  kapso learn import --archive gs://bucket/runs/rel-hm--user-churn/20260731T092629_lane-a2.tgz
"""
    )
    learn_sub = learn_parser.add_subparsers(dest="learn_command", required=True)

    learn_import = learn_sub.add_parser(
        "import",
        help="Import archived campaign tarballs into the trajectory store",
    )
    learn_import.add_argument("--subset", type=str, help="Subset YAML (trajectories: [{id, archive, role}])")
    learn_import.add_argument("--archive", type=str, help="One archive URI/path (.tgz)")
    learn_import.add_argument("--id", type=str, help="Trajectory id override for --archive")
    learn_import.add_argument("--config", type=str, default=None, help="Config path (default: packaged config.yaml)")
    learn_import.add_argument("--no-upload", action="store_true", help="Skip remote upload even if configured")

    learn_mine = learn_sub.add_parser(
        "mine",
        help="Mine store-resident trajectories into their derived mined/ views",
    )
    learn_mine.add_argument("--trajectory", type=str, help="One trajectory id")
    learn_mine.add_argument("--all", action="store_true", help="Mine every un-mined imported trajectory")
    learn_mine.add_argument("--force", action="store_true", help="Regenerate even if already mined")
    learn_mine.add_argument("--config", type=str, default=None, help="Config path (default: packaged config.yaml)")

    learn_grade = learn_sub.add_parser(
        "grade",
        help="Grade a bank against held-out trajectories (full) or one arriving trajectory (exam)",
    )
    learn_grade.add_argument("--bank", type=str, required=True, help="Bank checkout dir")
    learn_grade.add_argument("--bank-head", type=str, required=True, help="The graded bank head (lr_ tag or sha)")
    learn_grade.add_argument("--split", type=str, help="Split manifest (full mode)")
    learn_grade.add_argument("--learner-version", type=str, help="Learner version under exam (full mode)")
    learn_grade.add_argument("--trajectory", type=str, help="One trajectory id (exam mode)")
    learn_grade.add_argument("--config", type=str, default=None, help="Config path (default: packaged config.yaml)")

    learn_update = learn_sub.add_parser(
        "update",
        help="Run the update crew: fold a batch of mined+graded trajectories into the bank",
    )
    learn_update.add_argument("--batch-manifest", type=str,
                              help="YAML list of {trajectory, hindcast_report}; omit for docket-only consolidation")
    learn_update.add_argument("--learner-version", type=str, required=True, help="Crew version identifier")
    learn_update.add_argument("--split", type=str, help="Split manifest (development runs: asserts batch/held-out disjointness)")
    learn_update.add_argument("--config", type=str, default=None, help="Config path (default: packaged config.yaml)")

    learn_init_bank = learn_sub.add_parser(
        "init-bank", help="Create the bank home (bare repo + founding skeleton)"
    )
    learn_init_bank.add_argument("--config", type=str, default=None, help="Config path (default: packaged config.yaml)")

    learn_develop = learn_sub.add_parser(
        "develop",
        help="Run one learner version through the development regime (fresh bank, learn-set replay, held-out exam)",
    )
    learn_develop.add_argument("--split", type=str, required=True, help="Split manifest")
    learn_develop.add_argument("--learner-version", type=str, required=True, help="Crew version identifier")
    learn_develop.add_argument("--config", type=str, default=None, help="Config path (default: packaged config.yaml)")

    learn_codify = learn_sub.add_parser(
        "codify",
        help="Run one codify request (evolve minus ideation) against the production bank",
    )
    learn_codify.add_argument("--request", type=str, required=True, help="Request YAML (card, fixture, materials, gates)")
    learn_codify.add_argument("--config", type=str, default=None, help="Config path (default: packaged config.yaml)")

    learn_ingest = learn_sub.add_parser(
        "ingest",
        help="Operating-regime chain for one arriving campaign: mine -> exam -> lesson",
    )
    learn_ingest.add_argument("--trajectory", type=str, required=True, help="Trajectory id in the store")
    learn_ingest.add_argument("--learner-version", type=str, required=True, help="Crew version doing the lesson")
    learn_ingest.add_argument("--config", type=str, default=None, help="Config path (default: packaged config.yaml)")

    learn_gauntlet = learn_sub.add_parser(
        "gauntlet",
        help="Run the duplicate + stability traps against a completed development run",
    )
    learn_gauntlet.add_argument("--learner-version", type=str, required=True, help="Completed development run to trap")
    learn_gauntlet.add_argument("--config", type=str, default=None, help="Config path (default: packaged config.yaml)")

    learn_behave = learn_sub.add_parser(
        "behave",
        help="Run behavior scenarios (semantic production tests, agentic review)",
    )
    learn_behave.add_argument("--scenario", type=str, help="One scenario dir")
    learn_behave.add_argument("--all", action="store_true", help="Run every scenario")
    learn_behave.add_argument("--config", type=str, default=None, help="Config path (default: packaged config.yaml)")
    
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
    # WATCH command (observability design §3)
    # =========================================================================
    watch_parser = subparsers.add_parser(
        "watch",
        help="Watch a running evolve / learn / learn_knowledge operation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  kapso watch ./campaign                 # a workspace (-> .kapso/status.json)
  kapso watch learning/status            # newest status file in a directory
  kapso watch learning/status/learn-20260826T120000.json
  kapso watch ./campaign --follow        # live re-render until terminal
  kapso watch ./campaign --json | jq -r '[.state] | @tsv'
""",
    )
    watch_parser.add_argument(
        "path",
        help="Workspace, status file, or directory of status files",
    )
    watch_parser.add_argument(
        "--json", action="store_true",
        help="Print the status file once, as JSON",
    )
    watch_parser.add_argument(
        "--follow", action="store_true",
        help="Re-render on each heartbeat until the operation ends",
    )

    # =========================================================================
    # INBOX command (docs/research/evolve-hub-design.md v4 §4.6)
    # =========================================================================
    inbox_parser = subparsers.add_parser(
        "inbox",
        help="What evolve is waiting on you for, and how to answer it",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "A session that needs something only a person can provide asks "
            "through the inbox and the campaign pauses. `kapso inbox` shows "
            "what is waiting; `kapso inbox reply` answers one request and "
            "resumes the campaign here, continuing the session that asked."
        ),
        epilog="""
Examples:
  kapso inbox                              # every campaign waiting on you (this one, inside a campaign)
  kapso inbox ./campaign                   # this campaign's requests
  kapso inbox reply 1 "added the key"      # answer request 1 here; the campaign resumes
  kapso inbox reply "added the key"        # the id may be left out when one request is open
  kapso inbox reply ./campaign 1           # answer from elsewhere; an empty note means done
""",
    )
    inbox_parser.add_argument(
        "words", nargs="*",
        help="[CAMPAIGN]  or  reply [CAMPAIGN] [ID] [NOTE]",
    )
    inbox_parser.add_argument(
        "--yes", action="store_true",
        help="Store a reply that looks like a credential anyway",
    )
    inbox_parser.add_argument(
        "--config", type=str, default=None,
        help="Config whose inbox.registry lists the campaigns (default: packaged config.yaml)",
    )

    # =========================================================================
    # BANK command (lesson-bank sharing)
    # =========================================================================
    bank_parser = subparsers.add_parser(
        "bank",
        help="Manage the lesson bank's share remote",
        description=(
            "The bank is local-only by default. `connect` attaches a git "
            "remote as its origin and pushes the bank there; `create` "
            "first creates the GitHub repository via the gh CLI. "
            "Credentials stay git's/gh's own — kapso never handles them."
        ),
    )
    bank_sub = bank_parser.add_subparsers(dest="bank_command", required=True)
    bank_connect = bank_sub.add_parser(
        "connect",
        help="Attach a git remote as the bank's origin and push the bank",
    )
    bank_connect.add_argument(
        "url", type=str, help="Git URL of the bank repository (https or ssh)"
    )
    bank_connect.add_argument(
        "--config", type=str, default=None,
        help="Config path (default: packaged config.yaml)",
    )
    bank_create = bank_sub.add_parser(
        "create",
        help="Create a GitHub repo for the bank (gh CLI) and connect it",
    )
    bank_create.add_argument(
        "slug", type=str, help="org/name for the new repository"
    )
    bank_create.add_argument(
        "--public", action="store_true",
        help="Create the repository public (default: private)",
    )
    bank_create.add_argument(
        "--config", type=str, default=None,
        help="Config path (default: packaged config.yaml)",
    )

    # =========================================================================
    # DOCTOR command (onboarding)
    # =========================================================================
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check this machine can run what your config asks for",
        description=(
            "Reports the requirements the ACTIVE CONFIG implies — an "
            "all-codex config is never asked for claude, and a local "
            "codify target is never asked for gcloud. With no argument "
            "this is the union across all five verbs; name one "
            f"({', '.join(VERBS)}) to see just that verb's requirements, "
            "which are exactly the checks it runs at call time. Every "
            "missing item names the config key that wants it and one fix. "
            "Exits non-zero when a required item is missing. With "
            "--models, additionally fires a one-token live probe per "
            "{cli, model} pair — a subscription cap surfaces here in "
            "seconds instead of hours into a run."
        ),
    )
    doctor_parser.add_argument(
        "verb", nargs="?", choices=list(VERBS), default=None,
        help="Narrow the report to one verb's requirements",
    )
    doctor_parser.add_argument(
        "--models", action="store_true",
        help="Live-probe each configured {cli, model} pair (spends a few "
             "one-token calls)",
    )
    doctor_parser.add_argument(
        "--config", type=str, default=None,
        help="Config to check (default: packaged config.yaml)",
    )

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
    elif args.command == "watch":
        cmd_watch(args)
    elif args.command == "inbox":
        cmd_inbox(args)
    elif args.command == "bank":
        cmd_bank(args)
    elif args.command == "doctor":
        cmd_doctor(args)
    else:
        parser.print_help()
        print("\nError: Please specify a command (evolve, research, learn, deploy, index_kg, watch, inbox, bank, doctor)")
        sys.exit(1)


if __name__ == "__main__":
    main()
