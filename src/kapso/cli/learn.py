"""`kapso learn <campaign>`: learn from a finished campaign into your bank
(what `Kapso.learn()` does), with `--harness` for the harness the campaign
ran under when none is loaded by its record. The learning system's own
tools — the development regime (docs/research/learn-from-trajectories-design.md)
— live under `kapso learn dev <command>`."""

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import yaml

from kapso.cli.common import _detach, _without_detach_flags, learning_handoff
from kapso.kapso import DEFAULT_CONFIG_PATH, Kapso
from kapso.core.config import load_config
from kapso.execution.inbox import beside
from kapso.learning.corpus_import import import_archive, import_subset
from kapso.learning.behavior import BehaviorRunner
from kapso.learning.codify_run import CodifyRunDriver
from kapso.learning.develop import DevelopmentDriver
from kapso.learning.graders.frame import GradingFrame
from kapso.learning.graders.gauntlet import GauntletRunner
from kapso.learning.graders.split import assert_batch_disjoint, load_split, validate_split
from kapso.learning.update_frame import UpdateFrame
from kapso.learning.mining import MiningFrame
from kapso.learning.trajectory_store import TrajectoryStore

NAME = "learn"
DEV = "dev"
# The plain form's hidden subcommand: `kapso learn <campaign>` is rewritten
# to it before parsing (normalize_argv), so argparse sees one subcommand
# beside `dev` and the person never types it.
CAMPAIGN = "campaign"


def normalize_argv(argv: List[str]) -> List[str]:
    """`kapso learn <campaign> …` is the verb's own form: a first token
    after `learn` that is neither `dev` nor a flag is the campaign."""
    if len(argv) >= 2 and argv[0] == NAME and argv[1] not in (DEV, "-h", "--help") and not argv[1].startswith("-"):
        return [NAME, CAMPAIGN, *argv[1:]]
    return argv


def cmd_learn(args) -> None:
    """Handle the learn command: the campaign form, or the development
    regime's tools under `dev`."""
    if args.learn_command == CAMPAIGN:
        if args.detach:
            # Hours of crews: in a session of its own, the log beside the
            # campaign, the status under the learning status directory.
            campaign = Path(args.campaign).expanduser()
            _detach(
                "kapso learn", _without_detach_flags(sys.argv[1:]),
                log_path=beside(campaign, ".learn.log"), receipt_path=None,
                handoff=learning_handoff(load_config(args.config)["learning"]["status_dir"]),
            )
            return
        kapso = Kapso(config=args.config)
        if args.harness:
            kapso.load_harness(args.harness)
        lesson = kapso.learn(args.campaign)
        print(lesson.explain())
        return
    _dev(args, load_config(args.config))


def _dev(args, config) -> None:
    if args.dev_command == "import":
        store = TrajectoryStore.from_config(config)
        upload = False if args.no_upload else None
        if bool(args.subset) == bool(args.archive):
            print("Error: exactly one of --subset or --archive is required", file=sys.stderr)
            sys.exit(1)
        if args.subset:
            report_path = import_subset(
                store, args.subset, config["learning"]["import_report_dir"], upload
            )
            print(f"Import report: {report_path}")
        else:
            outcome = import_archive(store, args.archive, args.id, upload)
            print(f"{outcome['id']}: {outcome['status']}")
    elif args.dev_command == "mine":
        frame = MiningFrame.from_config(config)
        if bool(args.trajectory) == bool(args.all):
            print("Error: exactly one of --trajectory or --all is required", file=sys.stderr)
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
    elif args.dev_command == "grade":
        grading = GradingFrame(TrajectoryStore.from_config(config), config)
        run_root = config["learning"]["graders"]["run_root"]
        if bool(args.split) == bool(args.trajectory):
            print("Error: exactly one of --split (full) or --trajectory (exam) is required", file=sys.stderr)
            sys.exit(1)
        if args.split:
            if not args.learner_version:
                print("Error: --learner-version is required in full mode", file=sys.stderr)
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
    elif args.dev_command == "update":
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
    elif args.dev_command == "codify":
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
    elif args.dev_command == "ingest":
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
    elif args.dev_command == "gauntlet":
        runner = GauntletRunner(TrajectoryStore.from_config(config), config)
        verdict = runner.run(args.learner_version)
        root = Path(config["learning"]["develop"]["run_root"]).expanduser()
        print(f"Gauntlet: {root / args.learner_version / 'gauntlet.md'}")
        print(f"Rolled verdict: {verdict}")
    elif args.dev_command == "develop":
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
    elif args.dev_command == "behave":
        runner = BehaviorRunner(TrajectoryStore.from_config(config), config)
        run_root = config["learning"]["behavior"]["run_root"]
        if bool(args.scenario) == bool(args.all):
            print("Error: exactly one of --scenario or --all is required", file=sys.stderr)
            sys.exit(1)
        if args.scenario:
            result = runner.run_scenario(args.scenario, run_root)
            print(f"{result['scenario']}: {result['verdict']} — {result['rationale']}")
        else:
            rollup = runner.run_all(run_root)
            print(f"Behavior suite: {rollup['verdict']}")
            for row in rollup["scenarios"]:
                print(f"  {row['scenario']}: {row['verdict']}")


def add_parser(subparsers):
    learn_parser = subparsers.add_parser(
        "learn",
        help="Learn from a finished campaign into your bank; `dev` holds the learning system's own tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        epilog="""
Examples:
  kapso learn ./campaign                                     # import -> mine -> exam -> lesson, into your bank
  kapso learn ./campaign --harness ./harness                 # the harness the campaign ran under, when its record has none
  kapso learn ./campaign --detach                            # hours of crews in a session of its own; kapso watch learning/status --follow
  kapso learn dev import --subset docs/plans/learning/d1-subset.yaml
  kapso learn dev import --archive gs://bucket/runs/rel-hm--user-churn/20260731T092629_lane-a2.tgz
"""
    )
    learn_sub = learn_parser.add_subparsers(dest="learn_command", required=True, metavar="<campaign> | dev")
    # The plain form (never typed: normalize_argv rewrites `kapso learn
    # <campaign>` to it), kept off the docs coverage check's groups.
    learn_forms = learn_sub
    campaign = learn_forms.add_parser(CAMPAIGN, help=argparse.SUPPRESS)
    campaign.add_argument("campaign", type=str, help="The campaign directory (the solution's trace) or a store trajectory id")
    campaign.add_argument("--harness", type=str, default=None, help="The harness the campaign ran under (a name, a path or a git reference), when its record names none")
    campaign.add_argument("--config", type=str, default=None, help="Config path (default: packaged config.yaml)")
    campaign.add_argument(
        "--detach", action="store_true",
        help="Run the learn in a session of its own and return at once with its pid, log and watch line",
    )

    dev = learn_sub.add_parser(DEV, help="The learning system's own tools: the development regime")
    dev_sub = dev.add_subparsers(dest="dev_command", required=True)

    learn_import = dev_sub.add_parser(
        "import",
        help="Import archived campaign tarballs into the trajectory store",
    )
    learn_import.add_argument("--subset", type=str, help="Subset YAML (trajectories: [{id, archive, role}])")
    learn_import.add_argument("--archive", type=str, help="One archive URI/path (.tgz)")
    learn_import.add_argument("--id", type=str, help="Trajectory id override for --archive")
    learn_import.add_argument("--config", type=str, default=None, help="Config path (default: packaged config.yaml)")
    learn_import.add_argument("--no-upload", action="store_true", help="Skip remote upload even if configured")

    learn_mine = dev_sub.add_parser(
        "mine",
        help="Mine store-resident trajectories into their derived mined/ views",
    )
    learn_mine.add_argument("--trajectory", type=str, help="One trajectory id")
    learn_mine.add_argument("--all", action="store_true", help="Mine every un-mined imported trajectory")
    learn_mine.add_argument("--force", action="store_true", help="Regenerate even if already mined")
    learn_mine.add_argument("--config", type=str, default=None, help="Config path (default: packaged config.yaml)")

    learn_grade = dev_sub.add_parser(
        "grade",
        help="Grade a bank against held-out trajectories (full) or one arriving trajectory (exam)",
    )
    learn_grade.add_argument("--bank", type=str, required=True, help="Bank checkout dir")
    learn_grade.add_argument("--bank-head", type=str, required=True, help="The graded bank head (lr_ tag or sha)")
    learn_grade.add_argument("--split", type=str, help="Split manifest (full mode)")
    learn_grade.add_argument("--learner-version", type=str, help="Learner version under exam (full mode)")
    learn_grade.add_argument("--trajectory", type=str, help="One trajectory id (exam mode)")
    learn_grade.add_argument("--config", type=str, default=None, help="Config path (default: packaged config.yaml)")

    learn_update = dev_sub.add_parser(
        "update",
        help="Run the update crew: fold a batch of mined+graded trajectories into the bank",
    )
    learn_update.add_argument("--batch-manifest", type=str,
                              help="YAML list of {trajectory, hindcast_report}; omit for docket-only consolidation")
    learn_update.add_argument("--learner-version", type=str, required=True, help="Crew version identifier")
    learn_update.add_argument("--split", type=str, help="Split manifest (development runs: asserts batch/held-out disjointness)")
    learn_update.add_argument("--config", type=str, default=None, help="Config path (default: packaged config.yaml)")

    learn_develop = dev_sub.add_parser(
        "develop",
        help="Run one learner version through the development regime (fresh bank, learn-set replay, held-out exam)",
    )
    learn_develop.add_argument("--split", type=str, required=True, help="Split manifest")
    learn_develop.add_argument("--learner-version", type=str, required=True, help="Crew version identifier")
    learn_develop.add_argument("--config", type=str, default=None, help="Config path (default: packaged config.yaml)")

    learn_codify = dev_sub.add_parser(
        "codify",
        help="Run one codify request (build minus ideation) against the production bank",
    )
    learn_codify.add_argument("--request", type=str, required=True, help="Request YAML (card, fixture, materials, gates)")
    learn_codify.add_argument("--config", type=str, default=None, help="Config path (default: packaged config.yaml)")

    learn_ingest = dev_sub.add_parser(
        "ingest",
        help="Operating-regime chain for one arriving campaign: mine -> exam -> lesson",
    )
    learn_ingest.add_argument("--trajectory", type=str, required=True, help="Trajectory id in the store")
    learn_ingest.add_argument("--learner-version", type=str, required=True, help="Crew version doing the lesson")
    learn_ingest.add_argument("--config", type=str, default=None, help="Config path (default: packaged config.yaml)")

    learn_gauntlet = dev_sub.add_parser(
        "gauntlet",
        help="Run the duplicate + stability traps against a completed development run",
    )
    learn_gauntlet.add_argument("--learner-version", type=str, required=True, help="Completed development run to trap")
    learn_gauntlet.add_argument("--config", type=str, default=None, help="Config path (default: packaged config.yaml)")

    learn_behave = dev_sub.add_parser(
        "behave",
        help="Run behavior scenarios (semantic production tests, agentic review)",
    )
    learn_behave.add_argument("--scenario", type=str, help="One scenario dir")
    learn_behave.add_argument("--all", action="store_true", help="Run every scenario")
    learn_behave.add_argument("--config", type=str, default=None, help="Config path (default: packaged config.yaml)")
    return cmd_learn
