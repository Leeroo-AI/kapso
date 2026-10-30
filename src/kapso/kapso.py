# Kapso Agent - Main Entry Point
#
# The primary user-facing API for the Kapso Agent system: the verbs
# build, deploy and learn on one object.
#   Kapso.build() -> Kapso.deploy() -> Software.run(); Kapso.learn()
#
# Usage:
#     from kapso import Kapso, Brief, Budget, Solution
#
#     kapso = Kapso(config="./kapso-config.yaml")     # engine knobs only
#     solution = kapso.build(
#         Brief.read("kapso-brief.md"),
#         evaluation="./evaluation", data="./data", repo=".",
#         output="../churn-build", budget=Budget(minutes=480),
#     )
#     software = kapso.deploy(solution, target="local")
#     result = software.run({"input": "data"})
#     Solution.open("../churn-build")                  # the same solution, from any process

import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid

import git

import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Load environment variables from .env file (if present)
from dotenv import find_dotenv, load_dotenv
# Resolve .env from the caller's CWD upward. The no-arg default anchors at
# THIS file's directory — site-packages in a pip install — so the .env the
# README tells users to create next to their project was never read.
load_dotenv(find_dotenv(usecwd=True))

from kapso.execution.coding_agents.factory import CodingAgentFactory
from kapso.execution.observability import (
    mark_failed,
    KnowledgeStatus,
    LessonStatus,
    OperationStatusView,
    credits_summary,
    knowledge_summary,
    platform_cost_summary,
)
from kapso.execution.trace import TraceView, failed_runs, read_trace
from kapso.execution.notify import notify_build_end
from kapso.execution.orchestrator import OrchestratorAgent
from kapso.execution.evaluation_maintainer.registry import EvaluationRegistry
from kapso.execution.iteration_evaluator import IterationEvaluator
from kapso.handles import EXPERIENCE_RECORD_FIELDS, Brief, Budget, EvaluatorRef, ExperienceRef, Solution
from kapso.execution.evaluation_integrity import launch_evaluation_manifest
from kapso.environment.handlers.generic import GenericProblemHandler
from kapso.knowledge_base.search import KnowledgeSearchFactory
from kapso.knowledge_base.search.workflow_search import WorkflowRepoSearch
from kapso.core.agent_manifest import load_agent_manifest
from kapso.core.config import PACKAGED_CONFIG_PATH, is_packaged, load_config
from kapso.core.roles import validate_config
from kapso.experience.sources import experience_settings
from kapso.experience import (
    HOSTED,
    MINE,
    ExperienceError,
    ExperienceSource,
    ExperienceStore,
    bank_head,
    has_origin,
    probe_hosted_gate,
    parse_experience_reference,
    pull_home,
    recorded_reference,
    resolve_source,
)
from kapso.experience.ingest import (
    check_name,
    ingest,
    repository_source,
    research_sources,
    search_for,
)
from kapso.harness import (
    BUILD_MODULES,
    DEPLOY_MODULES,
    LEARN_MODULES,
    Harness,
    HarnessError,
    advisory_lines,
    load_harness,
    reload_harness,
    resume_refusal,
    unmet_message,
)
from kapso.harness import BUNDLED_ROOT
from kapso.harness.staging import host_for
from kapso.execution.inbox import (
    launch_receipt_path,
    Request,
    idea_line,
    inbox_path,
    launch_mismatches,
    load_requests,
    read_launch_record,
    record_reply,
    register_campaign,
    resume_arguments,
    write_launch_record,
)
from kapso.execution.run_checkpoint import (
    RunCheckpointIncompatibleError,
    RunCheckpointStore,
)
from kapso.core.preflight import run_preflight
from kapso.learning.graders.frame import GradingFrame
from kapso.learning.lesson_result import LessonResult, MemoryStatus
from kapso.learning.mining import MiningFrame
from kapso.learning.serving_launch import (
    plan_campaign_serving,
    stage_serving,
    task_coordinates,
)
from kapso.learning.trajectory_store import TrajectoryStore, save_trajectory
from kapso.learning.bank_remote import bank_origin, connect_bank
from kapso.learning.update_frame import UpdateFrame, init_bank

from kapso.deployment import DeployError, Software
from kapso.deployment.session import run_deployment
from kapso.deployment.targets import AUTO, resolve_target


# =============================================================================
# KAPSO AGENT
# =============================================================================

# Path to default configuration
DEFAULT_CONFIG_PATH = str(PACKAGED_CONFIG_PATH)


def _source_is_reachable(source: str) -> bool:
    """A recorded harness source that can be loaded again from here: a git
    form (a URL, `git+`, shorthand or scp-like with `@` or `#`), a
    directory, or a bundled name."""
    if "://" in source or source.startswith("git+") or "@" in source or "#" in source:
        return True
    return Path(source).expanduser().is_dir() or (BUNDLED_ROOT / source).is_dir()


def _same_reference(line: str, harness: Harness) -> bool:
    """Whether the brief's Harness line names the loaded harness: the same
    directory once resolved from the working directory, or the same
    reference as written."""
    candidate = Path(line).expanduser()
    directory = candidate if candidate.is_absolute() else Path.cwd() / candidate
    if directory.is_dir():
        return directory.resolve() == Path(harness.path).resolve()
    return line in (harness.source, harness.reference.raw)


def _absolute(path: Optional[str]) -> Optional[str]:
    """A launch argument as the launch record keeps it: resolved against
    the launch directory, so a resume or an inbox reply run from any other
    directory (a host session's, say) finds the same files."""
    return None if path is None else str(Path(path).expanduser().resolve())


def _git_rev(workspace: Path, ref: str) -> str:
    """The commit a ref names in the campaign workspace."""
    return subprocess.run(
        ["git", "-C", str(workspace), "rev-parse", ref],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _memory_overrides(serving_plan: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Per-campaign strategy params carrying what the loaded sources
    mount: a staged bank pulls in the `bank` gate, an index the
    wiki-search gates over its record, the hosted source the
    `leeroopedia` server. None is an independent config choice — the
    injected intro tells sessions those tools exist (E2E review
    2026-08-24: they did not).
    """
    if not serving_plan:
        return None
    overrides: Dict[str, Any] = {
        # The intro's path, read by the engine when it builds the
        # ideation and implementation prompts: staged after construction,
        # so a seeded workspace is still empty here.
        "experience_intro_path": serving_plan["intro_path"],
    }
    if serving_plan["bank_serving"]:
        overrides["bank_serving"] = serving_plan["bank_serving"]
    if serving_plan["hosted"]:
        overrides["hosted"] = serving_plan["hosted"]
    if serving_plan["kg_index_path"]:
        overrides["kg_index_path"] = serving_plan["kg_index_path"]
    return overrides

def _deliver(orchestrator, solve_result, *, brief: Brief, harness: Optional[Harness], experience) -> Solution:
    """The delivery, run inside the orchestrator's protected region: the
    strategy checks the workspace out at the delivered candidate's branch
    (None when nothing succeeded), the registry head says which evaluator
    measured it, and the record is written."""
    strategy = orchestrator.search_strategy
    campaign = Path(strategy.workspace.workspace_dir).resolve()
    best_branch = strategy.checkout_to_best_experiment_branch()
    head = EvaluationRegistry(str(campaign)).head()
    # The judge's words on the delivered candidate; with nothing
    # delivered, the last judgment says why (the third proof run
    # delivered node 0 and reported node 1's failed judgment).
    delivered = strategy.get_deliverable_experiment()
    feedback = (
        delivered.feedback if delivered is not None
        else solve_result.final_feedback.feedback
        if solve_result.final_feedback else None
    )
    solution = Solution(
        brief=brief,
        campaign=campaign,
        branch=best_branch,
        commit=_git_rev(campaign, "HEAD") if best_branch else None,
        base_commit=_git_rev(campaign, "main"),
        score=strategy.get_deliverable_score(),
        goal_achieved=bool(
            solve_result.final_feedback and solve_result.final_feedback.stop
        ),
        stopped_reason=solve_result.stopped_reason,
        stop_detail=solve_result.stop_detail,
        experiments=solve_result.iterations_run,
        experiments_total=solve_result.cumulative_iterations,
        cost_usd=float(solve_result.total_cost),
        # The inbox: what the person must answer when the campaign
        # paused for them (empty otherwise).
        requests=list(solve_result.requests),
        feedback=feedback,
        evaluator=(
            EvaluatorRef(
                version=head.version,
                evaluator_id=head.evaluator_id,
                provenance=head.provenance,
            )
            if head else None
        ),
        harness=harness.ref() if harness is not None else None,
        # What the campaign was served (wp3-experience.md): the loaded
        # sources, each at its pinned head and scope.
        experience=[source.ref() for source in experience],
        verification=delivered.verification if delivered is not None else None,
        rejected_candidates=sum(
            not node.evaluation_valid
            for node in strategy.get_experiment_history()
        ),
        callback_metrics=dict(
            getattr(solve_result.best_experiment, "metrics", {}) or {}
        ),
        callback_primary_metric=getattr(
            solve_result.best_experiment, "primary_metric", None
        ),
        # The sessions' runs that ended error, timeout or refused, from
        # their records (every adapter's `finally` has run by now).
        session_failures=failed_runs(campaign),
    )
    solution.save()
    return solution


class Kapso:
    """
    The main Kapso Agent class.

    A Kapso is an intelligent agent whose memory is the experience it is
    served (docs/plans/redesign/wp3-experience.md): what it measured by
    doing, earned via learn() into its own bank; what others measured, a
    bank you read; what others reported, the hosted gate or a corpus
    ingested into an index with add_experience(). build() serves exactly
    what is loaded, and every Solution records it.

    The closed loop:
        kapso = Kapso()
        kapso.add_experience(repo=url, name="org-x")      # once; then load it
        kapso.load_experience("org-x")
        solution = kapso.build(brief, budget=Budget(minutes=240))
        lesson   = kapso.learn(solution)     # import -> mine -> exam -> lesson
        kapso.load_experience("mine")
        solution2 = kapso.build(brief)       # served what the first campaign taught

    Deployment:
        software = kapso.deploy(solution)
        result = software.run({"ticker": "AAPL"})

    With an evaluation, data and a starting repository:
        solution = kapso.build(
            "Build a classifier with 95% accuracy",
            evaluation="./evaluation/",
            data="./datasets/",
            repo="https://github.com/owner/starter-repo",
        )
    """

    def __init__(
        self,
        config: Optional[str] = None,
        bank: Optional[str] = None,
    ):
        """
        Initialize a Kapso agent. Nothing starts here.

        Args:
            config: Path to the configuration file, the engine's knobs
                    (the packaged config.yaml when not given)
            bank: Your bank's home directory, where learn() writes and
                  `mine` is served from. Overrides the config default
                  (learning.bank.local_path).
        """
        # The override file (None: the packaged defaults alone) and the
        # effective config, the one every verb reads its blocks from.
        self.config_path = config
        self._config = load_config(config)
        validate_config(self._config, load_agent_manifest()["agents"])
        # The harness block (store, thresholds, the two session timeouts)
        # and the loaded harness: none until load_harness() is called.
        self._harness_settings = dict(self._config["harness"])
        self._harness: Optional[Harness] = None
        # By construction time every dependency import has settled, so
        # this atexit registration lands last (LIFO-first at exit) and
        # keeps third-party exit-time warning noise out of user output.
        CodingAgentFactory.register_quiet_exit()

        # The bank home resolves ONCE here: every later call (learn /
        # build) reads this resolution — never re-reading config at call
        # time.
        configured_bank = (
            (self._config.get("learning") or {}).get("bank") or {}
        ).get("local_path")
        bank_path = bank or configured_bank
        self._bank_home: Optional[Path] = (
            Path(bank_path).expanduser() if bank_path else None
        )
        # The experience homes; `mine` lives where the bank home is, which
        # `bank=` may have moved.
        self._experience_settings = experience_settings(self._config)
        if self._bank_home is not None:
            self._experience_settings["bank_home"] = str(self._bank_home)
        self._experience: List[ExperienceSource] = []
        # The knowledge search is the null search until a build loads an
        # index: it is connected from the index's record then, never from
        # the config alone (a build with nothing loaded needs no key).
        self.knowledge_search = KnowledgeSearchFactory.create_null()
        # Backend clients (Weaviate/Neo4j/embeddings) hold real sockets;
        # close them at interpreter exit so scripts end without unclosed-
        # socket ResourceWarnings (onboarding E2E finding #8).
        atexit.register(self.close)
        print("Initialized Kapso")

    # ------------------------------------------------------------ harness

    @property
    def harness(self) -> Optional[Harness]:
        """The loaded harness, or None: today's generic behaviour."""
        return self._harness

    def load_harness(self, ref: str) -> Harness:
        """Resolve a bundled name, a path or a git reference; run `check`
        and raise on a failure; hash it; read what this machine lacks into
        `harness.missing` without raising. A second call replaces the
        first: one object, one harness. Nothing starts here."""
        self._harness = load_harness(ref, settings=self._harness_settings)
        print(f"Harness: {self._harness.label()} from {self._harness.source}")
        for row in self._harness.missing:
            print(f"  needs {row.label}: {row.fix}  [{', '.join(row.modules)}]")
        return self._harness

    # --------------------------------------------------------- experience

    @property
    def experience(self) -> List[ExperienceSource]:
        """The loaded sources, in load order; empty is today's generic
        behaviour: nothing served."""
        return list(self._experience)

    def load_experience(self, ref: str) -> ExperienceSource:
        """Resolve `mine` (or `mine:all`, every card you measured whatever
        the harness), `leeroopedia`, a bank path, a bank git reference or
        an index name; confirm it exists and is readable; pin it as it is
        on disk (a bank at its head, an index at its id; loading never
        pulls, `update_experience` does); append it to the object's
        sources, replacing an earlier load of the same reference. Nothing
        is served until a verb runs."""
        reference = parse_experience_reference(ref, indexes_root=Path(self._experience_settings["indexes"]))
        source = resolve_source(reference, settings=self._experience_settings)
        self._experience = [s for s in self._experience if s.reference != source.reference] + [source]
        print(f"Experience: {source.label()}")
        return source

    def share_experience(self, url: str) -> None:
        """Attach `url` as your bank's remote and push the bank there; a
        bank that does not exist yet is created first. learn() pushes
        there from then on, `update_experience` pulls from there, and the
        intro says "measured by you and your team". Credentials are git's
        own."""
        home = Path(self._experience_settings["bank_home"]).expanduser()
        if not home.exists():
            init_bank(str(home))
            print(f"created your bank at {home}")
        connect_bank(home, url)
        print(f"your bank {home} shares to {url}")

    def add_experience(
        self,
        ref: Optional[str] = None,
        *,
        repo: Optional[str] = None,
        research: Optional[str] = None,
        name: Optional[str] = None,
    ) -> str:
        """Make a source that must exist before it can be loaded, and
        return its reference. `ref="mine"` creates your bank (a first
        learn() does too); `ref="leeroopedia"` checks the hosted gate's
        key is present. `repo=` ingests a repository, `research=` a
        question answered on the web, into a new index under `name`:
        the pages under the experience home, their own backend namespace,
        the record `index.json` that pins the index at its id. An index
        is never written again after this: a name that exists is refused
        (`remove_experience` first). Exactly one of the three is given."""
        given = [value for value in (ref, repo, research) if value]
        if len(given) != 1:
            raise ValueError(
                "add_experience takes exactly one of ref (`mine` or `leeroopedia`), repo= or research="
            )
        if ref == MINE:
            home = Path(self._experience_settings["bank_home"]).expanduser()
            if home.exists():
                print(f"your bank exists at {home}")
            else:
                init_bank(str(home))
                print(f"created your bank at {home} (empty; learn() fills it, `mine` serves it)")
            return MINE
        if ref == HOSTED:
            server = probe_hosted_gate(self._experience_settings)
            print(f"{HOSTED}: its key is present and the gate answers as {server}; `{HOSTED}` loads it")
            return HOSTED
        if ref is not None:
            raise ValueError(
                f"`{ref}` is not something to add: `mine` or `leeroopedia` by reference, a repository with "
                "repo=, a question with research=; a bank by path or git reference needs no add"
            )
        store = ExperienceStore(self._experience_settings["banks"], self._experience_settings["indexes"])
        name = check_name(name, store)
        run_preflight("experience", self._config, research=research is not None)
        sources = (
            repository_source(repo) if repo
            else research_sources(research, self._experience_settings, self.config_path)
        )
        heartbeat_seconds = self._status_heartbeat_seconds()
        status = KnowledgeStatus(
            self._status_file("experience"),
            heartbeat_seconds=heartbeat_seconds,
            daemon=bool(heartbeat_seconds),
        )
        print(f"status: {status.path}")
        try:
            record, result = ingest(
                sources, name=name, origin=repo or research, store=store,
                config=self._config, config_path=self.config_path, status=status,
            )
        finally:
            in_flight = sys.exc_info()[1]
            if in_flight is not None:
                status.failed(in_flight)
        status.done(
            pages_extracted=result.total_pages_extracted, created=result.created,
            edited=result.edited, errors=len(result.errors),
        )
        print(f"Experience added: {name} (index {record.id}, {record.page_count} pages); `{name}` loads it")
        return name

    def update_experience(self, ref: Optional[str] = None) -> None:
        """The network step loading never takes: `mine` takes its remote's
        main (fast-forward; a history that does not is an error in git's
        words), a bank you read at a moving reference is fetched again,
        a tag or a commit never moves. `None` updates every bank on this
        machine. A campaign keeps the pin it launched with whatever moves
        here; an index is what `add` made (remove it and add it again)."""
        store = ExperienceStore(self._experience_settings["banks"], self._experience_settings["indexes"])
        if ref is None:
            references = [parse_experience_reference(MINE, indexes_root=Path(self._experience_settings["indexes"]))]
            references += [
                parse_experience_reference(
                    f"git+{bank['remote']}@{bank['rev']}", indexes_root=Path(self._experience_settings["indexes"]),
                )
                for bank in store.banks()
            ]
        else:
            references = [parse_experience_reference(ref, indexes_root=Path(self._experience_settings["indexes"]))]
        for reference in references:
            if reference.kind == MINE:
                home = Path(self._experience_settings["bank_home"]).expanduser()
                if not home.exists():
                    print(f"{MINE}: no bank yet (`kapso experience add {MINE}`)")
                elif not has_origin(home):
                    print(f"{MINE}: no remote (`kapso experience share <url>`), nothing to pull")
                else:
                    before = bank_head(home)
                    pull_home(home)
                    after = bank_head(home)
                    print(f"{MINE}: {'at ' + after[:12] + ', up to date' if after == before else f'now at {after[:12]} (was {before[:12]})'}")
            elif reference.kind == "git":
                moved = store.update(reference)
                if moved["before"] == moved["after"]:
                    print(f"{reference.raw}: at {moved['after'][:12]}: a tag or a commit never moves")
                else:
                    print(f"{reference.raw}: now at {moved['after'][:12]} (was {moved['before'][:12]})")
            else:
                raise ExperienceError(
                    f"`{reference.raw}` has nothing to update: the hosted gate and a bank by path are read in place, "
                    "and an index is what `add` made (`remove` it and add it again)"
                )

    def remove_experience(self, ref: str) -> None:
        """Delete what `load` fetched or `add` made: a bank you read by git
        reference (its clone and record) or an index by name (its backend
        namespace, its pages and its record). Your bank, the hosted gate
        and a bank by path are not the store's to remove."""
        store = ExperienceStore(self._experience_settings["banks"], self._experience_settings["indexes"])
        reference = parse_experience_reference(ref, indexes_root=Path(self._experience_settings["indexes"]))
        if reference.kind == "index":
            record = store.index_record(reference.name)
            search = search_for(record, self._config)
            search.clear()
            search.close()
            print(f"removed {store.remove_index(reference.name)} (index {record.id}; its collection and corpus cleared)")
        elif reference.kind == "git":
            print(f"removed {store.remove_bank(reference)}")
        else:
            raise ExperienceError(
                f"`{ref}` is not something to remove: your bank is yours (delete its directory yourself), "
                "the hosted gate is a key in .env, a bank by path is read in place"
            )
        self._experience = [source for source in self._experience if source.reference != reference.raw]

    # =========================================================================
    # Knowledge search
    # =========================================================================
    
    def _close_knowledge_search(self) -> None:
        # Runs at atexit — tolerate a test-stubbed backend without close();
        # every real KnowledgeSearch implements it (base class contract).
        closer = getattr(self.__dict__.get("knowledge_search"), "close", None)
        if callable(closer):
            closer()

    def close(self) -> None:
        """Release the knowledge-search backend's network clients
        (Weaviate, Neo4j, embeddings). Idempotent — registered atexit at
        construction so scripts end without unclosed-socket
        ResourceWarnings; also safe to call earlier when done with
        knowledge search (a build with an index loaded connects fresh)."""
        self._close_knowledge_search()

    # =========================================================================
    # Learning from experience (learn-from-trajectories)
    # =========================================================================

    def _bank_head(self) -> str:
        """Current head of the bank home (fail loud when absent)."""
        return subprocess.run(
            ["git", "--git-dir", str(self._bank_home), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

    @property
    def memory(self) -> MemoryStatus:
        """Your own bank right now: its head, its active cards, the
        trajectories in the store (what is loaded for serving is
        `experience`)."""
        bank_head = None
        active_cards = None
        store_count = None
        if self._bank_home is not None and self._bank_home.exists():
            bank_head = self._bank_head()
            listing = subprocess.run(
                ["git", "--git-dir", str(self._bank_home), "ls-tree",
                 "-r", "--name-only", "HEAD"],
                check=True, capture_output=True, text=True,
            ).stdout.splitlines()
            active_cards = sum(
                1 for path in listing
                if (path.startswith("insights/") or path.startswith("procedures/"))
                and path.endswith(".md") and not path.endswith("index.md")
            )
            store = TrajectoryStore.from_config(self._config)
            store_count = len(store.list_manifests())
        return MemoryStatus(
            bank_path=str(self._bank_home) if self._bank_home else None,
            bank_head=bank_head,
            bank_active_cards=active_cards,
            store_trajectories=store_count,
        )

    # =========================================================================
    # Observability (evolve-observability-design.md v3)
    # =========================================================================

    @staticmethod
    def status(path: str) -> OperationStatusView:
        """Read any operation's status file — a workspace, a status file,
        or a directory of them. A classmethod on purpose: observing a run
        needs no constructed agent. `view.explain()` renders the same
        screen `kapso watch` shows."""
        return OperationStatusView(path)

    @staticmethod
    def trace(path: str) -> TraceView:
        """The campaign's record read after the fact: the attempts with
        their sessions and run entries, the lessons served and used, what
        was voided or rejected and on what evidence, the sessions that
        failed, the delivered candidate. `view.explain()` is the text
        `kapso trace` prints, `view.data` what `--json` prints; a campaign
        still running renders partial. A pure reader."""
        return read_trace(path)

    def _status_heartbeat_seconds(self) -> Optional[float]:
        """The status daemons' beat, from the config's budget block (the
        same key that paces a build's durable clock)."""
        interval = self._config["budget"]["checkpoint_heartbeat_seconds"]
        return float(interval) if interval else None

    def _status_file(self, operation: str) -> Path:
        status_dir = (self._config.get("learning") or {}).get("status_dir")
        if status_dir is None:
            # Sourced from the canonical config, never re-hardcoded
            # (Rule 1): caller configs that omit the key inherit the
            # packaged default from its single home.
            status_dir = self._config["learning"]["status_dir"]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        return Path(status_dir).expanduser() / f"{operation}-{stamp}.json"

    def learn(
        self,
        source: Union[Solution, str],
        *,
        trajectory_id: Optional[str] = None,
        learner_version: Optional[str] = None,
        exam: bool = True,
        push: Optional[bool] = None,
        on_status: Optional[Any] = None,
    ) -> LessonResult:
        """Learn from one finished campaign: import it into the trajectory
        store, mine it, grade the bank on it (exam-before-lesson), then run
        the update crew — evidence-priced cards, one tagged bank commit.

        Args:
            source: What to learn from —
                - the Solution returned by build() (its campaign
                  directory, `trace`, is read directly),
                - a path to a campaign directory (archived + imported), or
                - a trajectory id already in the store (import skipped).
            trajectory_id: Store id when importing a directory; default is
                derived (<goal-or-dir-slug>/<UTC-stamp>).
            learner_version: Update-crew version; default from config
                (learning.update_crew.default_version).
            exam: Run the hindcast exam against the pinned pre-lesson bank
                head first. False is for development replays only.
            push: Push the bank commit to the bank's `origin` remote
                (attached via `kapso experience share <url>`). None means
                "push exactly when an origin is attached".
            on_status: Optional hook called with the status dict after
                every status write and heartbeat (observability §4b).
                Exceptions from the hook propagate.

        Returns:
            LessonResult — what changed in the bank and the paper trail.
        """
        started = datetime.now(timezone.utc)
        if self._bank_home is None:
            raise ValueError(
                "no bank configured — set learning.bank.local_path or "
                "pass Kapso(bank=...)"
            )
        if not self._bank_home.exists():
            init_bank(str(self._bank_home))
            print(f"initialized bank home at {self._bank_home}")
        origin = bank_origin(self._bank_home)
        should_push = bool(origin) if push is None else push
        # Preflight (onboarding E2E findings #1 and #5): every crew's CLI
        # and credentials, plus the bank's push destination, verified in
        # seconds up front — never after hours of crew work on the far side.
        run_preflight(
            "learn", self._config,
            bank_home=self._bank_home, push=should_push,
        )
        learning_config = self._config["learning"]
        version = (
            learner_version
            or learning_config["update_crew"]["default_version"]
        )
        store = TrajectoryStore.from_config(self._config)

        # Observability (design §2): crew sessions run 30+ minutes between
        # updates, so the base-class daemon carries liveness.
        heartbeat_seconds = self._status_heartbeat_seconds()
        status = LessonStatus(
            self._status_file("learn"),
            heartbeat_seconds=heartbeat_seconds,
            daemon=bool(heartbeat_seconds),
            on_status=on_status,
        )
        print(f"status: {status.path}")
        try:
            return self._learn_chain(
                source, trajectory_id, version, exam, should_push, origin,
                started, store, learning_config, status,
            )
        finally:
            in_flight = sys.exc_info()[1]
            if in_flight is not None:
                status.failed(in_flight)

    def _learn_chain(
        self,
        source: Union[Solution, str],
        trajectory_id: Optional[str],
        version: str,
        exam: bool,
        should_push: bool,
        origin: Optional[str],
        started: datetime,
        store: TrajectoryStore,
        learning_config: Dict[str, Any],
        status: LessonStatus,
    ) -> LessonResult:
        status.phase("harvest")

        # The harness for the learning sessions: the build's (or the loaded
        # one, which must share its name) for a Solution; the loaded one, if
        # any, for a path or a trajectory id. Its learning module is staged
        # into the crews' sessions; the trajectory records which harness learned.
        # A campaign directory with a record is the same case as its Solution.
        record = source if isinstance(source, Solution) else None
        if record is None and isinstance(source, str) and (Path(source) / ".kapso" / "solution.json").is_file():
            record = Solution.open(source)
        if record is not None:
            learning_harness = self._harness_for(record, verb="learn")
        elif self._harness is not None:
            learning_harness = reload_harness(self._harness, settings=self._harness_settings)
        else:
            learning_harness = None

        # --- dispatch (design §2) ---
        if isinstance(source, Solution):
            campaign_dir = source.campaign
            slug_seed = source.brief.objective
        else:
            known_ids = {m["id"] for m in store.list_manifests()}
            if source in known_ids:
                campaign_dir = None
                trajectory_id = source
                slug_seed = source
            else:
                campaign_dir = Path(source).expanduser()
                slug_seed = campaign_dir.name
                if not campaign_dir.is_dir():
                    raise FileNotFoundError(
                        f"learn() source {source!r} is neither a store "
                        f"trajectory id nor an existing campaign directory"
                    )

        if campaign_dir is not None:
            if trajectory_id is None:
                slug = "".join(
                    ch if ch.isalnum() else "-" for ch in slug_seed.lower()
                ).strip("-")[:48].strip("-") or "campaign"
                stamp = started.strftime("%Y%m%dT%H%M%S")
                trajectory_id = f"{slug}/{stamp}_facade"
            # Harvest bridge (the same step benchmark drivers run at
            # campaign completion). A bare build workspace lacks the
            # benchmark-shaped artifacts, so the facade synthesizes the
            # historical-contract trio HONESTLY from what the campaign
            # actually produced — each generated file says so.
            kapso_dir = campaign_dir / ".kapso"
            kapso_dir.mkdir(exist_ok=True)
            meta_path = campaign_dir / "campaign_meta.json"
            if not meta_path.is_file():
                meta_path.write_text(json.dumps({
                    "goal": slug_seed,
                    "workspace": str(campaign_dir),
                    "harvested_by": "Kapso.learn",
                    "harvested_at": started.isoformat(timespec="seconds"),
                }, indent=1))
            report_path = campaign_dir / "final_report.json"
            if not report_path.is_file():
                # The outcome from the solution's record, whether the
                # object or the campaign directory's own was given.
                report: Dict[str, Any] = {"generated_by": "Kapso.learn"}
                if record is not None:
                    report.update({
                        "goal": record.brief.objective,
                        "goal_achieved": record.goal_achieved,
                        "final_score": record.score,
                        "stopped_reason": record.stopped_reason,
                        "experiments": record.experiments,
                        "experiments_total": record.experiments_total,
                        "cost_usd": record.cost_usd,
                    })
                report_path.write_text(json.dumps(report, indent=1, default=str))
            log_path = campaign_dir / ".kapso" / "campaign.log"
            if not log_path.is_file():
                lines = ["# facade-generated harvest log (Kapso.learn)"]
                if record is not None:
                    lines.append(record.explain())
                log_path.write_text("\n".join(lines) + "\n")
            # The store contract requires registered runs. A bare build
            # workspace records its experiments in the history store, not
            # a runs/ archive — synthesize run manifests from that REAL
            # history (one per recorded experiment; a campaign with zero
            # experiments has nothing to learn from and fails loud).
            runs_dir = campaign_dir / "runs"
            if not any(runs_dir.glob("run_*")):
                history_path = (
                    campaign_dir / ".kapso" / "experiment_history.json"
                )
                if not history_path.is_file():
                    raise FileNotFoundError(
                        f"learn() source {campaign_dir} has neither runs/ "
                        "nor .kapso/experiment_history.json — nothing to "
                        "learn from"
                    )
                history = json.loads(history_path.read_text())
                entries = (
                    history if isinstance(history, list)
                    else history.get("experiments", [])
                )
                if not entries:
                    raise ValueError(
                        f"learn() source {campaign_dir} recorded no "
                        "experiments — nothing to learn from"
                    )
                for index, entry in enumerate(entries, start=1):
                    run_dir = runs_dir / f"run_{index:04d}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    (run_dir / "manifest.txt").write_text(
                        json.dumps(
                            {"generated_by": "Kapso.learn", **entry}
                            if isinstance(entry, dict)
                            else {"generated_by": "Kapso.learn",
                                  "experiment": entry},
                            indent=1, default=str,
                        )
                    )
            serving_record = (
                campaign_dir / ".kapso" / "serving" / "serving-record.yaml"
            )
            # The sources the campaign was served, each at its pinned
            # head, in the records' own shape (the serving record adds the
            # count served); the launcher wrote them, sessions never stamp
            # anything. Unserved: the empty list, as every other record.
            experience: List[Dict[str, Any]] = []
            if serving_record.is_file():
                experience = [
                    vars(ExperienceRef(**{key: source[key] for key in EXPERIENCE_RECORD_FIELDS}))
                    for source in yaml.safe_load(serving_record.read_text())["sources"]
                ]
            save_trajectory(
                store,
                trajectory_id,
                work_dir=str(campaign_dir),
                campaign_log=str(log_path),
                contract="historical",
                # A build workspace IS a git repo, and `.git/index` is
                # rewritten by any git command a later reader runs — the
                # mining frame's raw-immutability check then fails the view
                # ("raw file .git/index was modified during mining", found
                # live 2026-08-24). The repo's own object store is not
                # campaign evidence: the working tree, living documents,
                # run manifests and logs are.
                work_dir_exclude=(".git",),
                experience=experience,
                # The build's harness, what lessons are scoped by; and the
                # one that did the learning when it is another version.
                harness=vars(record.harness) if record is not None and record.harness else None,
                learning_harness=(
                    learning_harness.ref_dict()
                    if learning_harness is not None and (
                        record is None or record.harness is None
                        or learning_harness.hash != record.harness.hash
                    ) else None
                ),
                upload=None,
            )
            print(f"  Harvested into store: {trajectory_id}")
            status.note(
                f"harvested into store ({trajectory_id})",
                trajectory_id=trajectory_id,
            )

        # --- mine (idempotent: skip when the view exists) ---
        status.phase("mine", trajectory_id=trajectory_id)
        manifest = store.manifest(trajectory_id)
        if not manifest.get("derived", {}).get("mined"):
            mined_dir = MiningFrame.from_config(self._config).mine(
                trajectory_id, harness=learning_harness, status=status,
            )
            print(f"  Mined view: {mined_dir}")
            status.note(f"mined view written ({mined_dir})")

        # --- the exam pin: clone + record the head BEFORE the lesson ---
        graders_root = Path(
            learning_config["graders"]["run_root"]
        ).expanduser()
        checkout = (
            graders_root / "ingest-serving"
            / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
               + "-" + uuid.uuid4().hex[:8])
        )
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--quiet", str(self._bank_home), str(checkout)],
            check=True,
        )
        bank_head_before = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

        status.note(
            f"pinned pre-lesson bank head {bank_head_before[:8]}",
            bank_head_before=bank_head_before,
        )
        exam_report_path: Optional[str] = None
        if exam:
            status.phase("exam")
            grading = GradingFrame(store, self._config)
            learn_set_ids = [
                row["id"] for row in store.list_manifests()
                if row["id"] != trajectory_id
                and (store.local / row["id"] / "mined").is_dir()
            ]
            exam_report_path = str(grading.grade_exam(
                trajectory_id, str(checkout), bank_head_before,
                str(graders_root), learn_set_ids,
            ))
            print(f"  Exam report: {exam_report_path}")
            status.note(f"exam graded ({exam_report_path})")

        # --- the lesson ---
        status.phase("lesson")
        frame = UpdateFrame(store, self._config, bank_home=self._bank_home)
        run_dir = frame.run_update(
            [{"trajectory": trajectory_id,
              "hindcast_report": exam_report_path}]
            if exam_report_path else
            [{"trajectory": trajectory_id}],
            learning_config["update_crew"]["run_root"],
            version,
            harness=learning_harness,
        )
        bank_head_after = self._bank_head()

        # --- what changed (derived from the bank itself, never trusted) ---
        cards_created: List[str] = []
        cards_updated: List[str] = []
        if bank_head_after != bank_head_before:
            diff = subprocess.run(
                ["git", "--git-dir", str(self._bank_home), "diff",
                 "--name-status", bank_head_before, bank_head_after,
                 "--", "insights", "procedures"],
                check=True, capture_output=True, text=True,
            ).stdout.splitlines()
            for line in diff:
                change_status, _, path = line.partition("\t")
                if not path.endswith(".md") or path.endswith("index.md"):
                    continue
                # insights/<name>.md -> stem; procedures/<name>/card.md
                # -> the directory name (the card's identity).
                parts = Path(path).parts
                name = (
                    Path(path).parent.name
                    if parts[0] == "procedures" and parts[-1] == "card.md"
                    else Path(path).stem
                )
                if change_status.startswith("A"):
                    cards_created.append(name)
                elif change_status.startswith("M"):
                    cards_updated.append(name)

        admitted = bank_head_after != bank_head_before
        status.note(
            f"lesson landed: {len(cards_created)} created, "
            f"{len(cards_updated)} updated; bank "
            f"{bank_head_before[:8]} -> {bank_head_after[:8]}"
            if admitted else
            f"lesson admitted nothing: the bank stays at {bank_head_before[:8]}"
        )

        # --- push (to the bank's own origin; preflighted at learn() start) ---
        status.phase("push")
        pushed = False
        if should_push and admitted:
            subprocess.run(
                ["git", "--git-dir", str(self._bank_home), "push", "origin",
                 "main", "--tags"],
                check=True,
            )
            pushed = True

        duration = (datetime.now(timezone.utc) - started).total_seconds()
        status.done(
            cards={
                "created": sorted(cards_created),
                "updated": sorted(cards_updated),
            },
            bank_head_after=bank_head_after,
            pushed=pushed,
        )
        if not admitted:
            print(f"lesson admitted nothing: the bank stays at {bank_head_before[:12]} (the report says why)")
        elif pushed:
            print(f"lesson banked and pushed to {origin}")
        else:
            print("lesson banked locally · to share: kapso experience share <url>")
        return LessonResult(
            trajectory_id=trajectory_id,
            bank_head_before=bank_head_before,
            bank_head_after=bank_head_after,
            cards_created=sorted(cards_created),
            cards_updated=sorted(cards_updated),
            exam_report_path=exam_report_path,
            lesson_report_path=str(Path(run_dir) / "report.md"),
            metadata={
                "learner_version": version,
                "pushed": pushed,
                "duration_minutes": round(duration / 60, 1),
                "status_path": str(status.path),
            },
        )

    def _launch_config_path(self) -> Optional[str]:
        """The config as the launch record keeps it: None for the packaged
        defaults, however they were named, so a resume from another
        install compares equal; else the override file's absolute path."""
        return None if is_packaged(self.config_path) else _absolute(self.config_path)

    def build(
        self,
        brief: Union[str, Brief, None] = None,
        *,
        evaluation: Optional[str] = None,
        data: Optional[str] = None,
        repo: Optional[str] = None,
        output: Optional[str] = None,
        budget: Optional[Budget] = None,
        resume: bool = False,
        on_status: Optional[Any] = None,
        iteration_evaluator: Optional[IterationEvaluator] = None,
        iteration_evaluator_failure_policy: str = "record",
        coding_agent: Optional[str] = None,
    ) -> Solution:
        """Build software that meets the brief, by experiment: propose a
        candidate, implement it on its own branch, measure it, keep the
        best, repeat until the goal is met or the budget is spent.

        Args:
            brief: What to build: a sentence, or a `Brief` with the metric,
                the baseline, the target and the rules. A string becomes a
                brief with only an objective. The whole brief reaches the
                engine, never clipped. Left out only on a resume, which
                takes it from the checkpoint.
            evaluation: Directory holding the evaluation; copied to
                `kapso_evaluation/` in the workspace and integrity-protected,
                so a candidate that edits it is rejected.
            data: Directory holding the data; copied to `kapso_datasets/`.
            repo: The starting repository: a local path (copied, never
                modified) or a GitHub URL (cloned). None starts empty, or
                from a workflow the knowledge graph knows.
            output: The campaign directory, the solution's `trace`. A
                fresh campaign needs it empty; a resume needs it to hold
                the campaign.
            budget: What the build may spend. Its set fields win over the
                brief's budget; on a resume the launch's values fill what
                is still unset; the config's defaults fill the rest.
            resume: Continue the campaign in `output`: its workspace and a
                compatible checkpoint must exist. Settings the checkpoint
                fingerprints (coding agent, evaluation, config, experience)
                must match the launch's; a disagreement is refused by name.
            on_status: Called with the status dict after every status
                write and heartbeat.
            iteration_evaluator: Callback that evaluates each finalized
                candidate in an isolated detached worktree;
                `iteration_evaluator_failure_policy` is `record` (store
                the error on the candidate) or `raise` (stop the run).

        Returns:
            The `Solution`: the delivered candidate, its measurement and
            the account of the campaign. It is also written to
            `<output>/.kapso/solution.json`, so `Solution.open(output)`
            reads it from any process.
        """
        # To the second, as the status file stamps its own start.
        launched_at = datetime.now(timezone.utc).replace(microsecond=0)
        if isinstance(brief, str):
            brief = Brief(objective=brief)
        elif brief is not None and not isinstance(brief, Brief):
            raise TypeError(
                f"brief must be a string or a Brief, got {type(brief).__name__}"
            )
        if brief is None and not resume:
            raise ValueError(
                "build() needs a brief: a sentence or a Brief; only a resume "
                "may leave it out, and then it comes from the checkpoint"
            )
        # Absolute from here on: the workspace, its sessions and the record
        # are addressed from directories other than the launch's (a session
        # clone commits its run data by path, a resume runs from anywhere),
        # and a relative `output` broke the first session close on the
        # fresh-machine proof run. A repository URL is left as it is.
        eval_dir, data_dir, output_path = _absolute(evaluation), _absolute(data), _absolute(output)
        initial_repo = (
            repo if repo is None or self._is_github_url(repo) else _absolute(repo)
        )
        for label, directory in (("evaluation", eval_dir), ("data", data_dir)):
            if directory is not None and not Path(directory).is_dir():
                raise FileNotFoundError(f"{label} directory not found: {directory}")
        if initial_repo is not None and not self._is_github_url(initial_repo) \
                and not Path(initial_repo).is_dir():
            raise FileNotFoundError(f"repo directory not found: {initial_repo}")
        checkpoint = None
        record = None
        if resume:
            self._validate_resume_workspace(output_path)
            # A resume without a checkpoint is refused by the orchestrator
            # with its own message; here the checkpoint is read when it
            # exists, for the brief and the harness comparison.
            store = RunCheckpointStore(output_path)
            checkpoint = store.load() if store.exists() else None
            if brief is None:
                if checkpoint is None:
                    checkpoint = store.load()
            if brief is None:
                # The checkpoint's goal is the brief as the launch rendered
                # it, so the one grammar reads it back; it comes first so
                # its budget takes part in the merge below.
                brief = Brief.parse(
                    checkpoint.goal,
                    source=str(Path(output_path) / ".kapso" / "run_state.json"),
                )
        spend = (budget or Budget()).merged_over(brief.budget)
        max_iterations = spend.experiments
        time_budget_minutes = spend.minutes
        cost_budget = spend.usd
        finalization_reserve_minutes = spend.reserve_minutes
        if resume:
            # A resume needs the launch's own arguments. The launch record
            # holds them, so a bare `kapso build --output <dir> --resume`
            # works; anything passed explicitly must agree with the record
            # where the checkpoint fingerprints it, and a disagreement is
            # named here rather than surfacing as a bare fingerprint
            # mismatch from the checkpoint.
            record = read_launch_record(output_path)
            if record is not None:
                # The campaign's own .env, from wherever the resume runs;
                # the process environment keeps precedence.
                if record.get("dotenv_path"):
                    load_dotenv(record["dotenv_path"])
                arguments = resume_arguments(record, {
                    "host": coding_agent,
                    "eval_dir": eval_dir,
                    "data_dir": data_dir,
                    "experiments": max_iterations,
                    "minutes": time_budget_minutes,
                    "usd": cost_budget,
                    "reserve_minutes": finalization_reserve_minutes,
                })
                # The loaded experience is campaign identity: a resume with
                # nothing loaded takes the record's list and re-pins each
                # source at its recorded head and scope; one that names a
                # different list is refused by name below.
                if not self._experience and record.get("experience"):
                    for recorded in record["experience"]:
                        # Re-loaded from where the launch recorded it, and
                        # named as the launch named it (a bank read by a
                        # relative path keeps that spelling in every record).
                        self.load_experience(recorded_reference(recorded)).reference = recorded["reference"]
                for source in self._experience:
                    recorded = next(
                        (r for r in record.get("experience", []) if r["reference"] == source.reference), None,
                    )
                    if recorded is not None and source.kind == "index" and source.head != recorded["head"]:
                        raise ExperienceError(
                            f"{source.reference} was re-ingested since this campaign launched "
                            f"(index {recorded['head']} then, {source.head} now); a campaign is served "
                            "the corpus it started with — start a new one to use the new index"
                        )
                    if recorded is not None and source.kind == "bank":
                        source.head = recorded["head"]
                mismatches = launch_mismatches(record, {
                    **arguments,
                    "config_path": self._launch_config_path(),
                    "experience": [vars(source.ref()) for source in self._experience],
                })
                if mismatches:
                    changed = "; ".join(
                        f"{key} was {was!r}, now {now!r}"
                        for key, (was, now) in mismatches.items()
                    )
                    raise RunCheckpointIncompatibleError(
                        "This resume changes settings the checkpoint "
                        f"fingerprints: {changed}. Pass the launch's values, or "
                        "start a new campaign in a new output path."
                    )
                coding_agent = arguments["host"]
                eval_dir = arguments["eval_dir"]
                data_dir = arguments["data_dir"]
                max_iterations = arguments["experiments"]
                time_budget_minutes = arguments["minutes"]
                cost_budget = arguments["usd"]
                finalization_reserve_minutes = arguments["reserve_minutes"]
        # The whole brief is the engine's goal text: the judge reads every
        # field, and the checkpoint fingerprints these exact bytes.
        goal_text = brief.render()
        if max_iterations is None:
            max_iterations = int(self._config["budget"]["experiments"])
        # A launch with an output directory is registered before the launch
        # checks: a campaign that dies at one (a refused evaluation suite,
        # a failed preflight) is still one `kapso watch --all` lists,
        # stopped, with its receipt's log (wp6-observability.md, change 8).
        # A resume was registered by its launch; a build with no output
        # directory is registered once the orchestrator has made one.
        inbox_block = self._config["inbox"]
        registered = output_path is not None and not resume
        if registered:
            register_campaign(inbox_block["registry"], output_path, brief.objective)
        if eval_dir:
            # Validate caller-owned evaluation inputs before resolving an
            # initial repository or initializing the experiment workspace
            # (a result.json an earlier run left there is refused here).
            launch_evaluation_manifest(eval_dir)
        # Preflight before the first session is spawned: the campaign's
        # own CLIs and credentials, the gates the strategy names, and the
        # stores this agent's knowledge/experience actually connect to.
        run_preflight(
            "build", self._config,
            coding_agent=coding_agent,
            index=next((source.reference for source in self._experience if source.kind == "index"), None),
        )

        print(f"\n{'='*60}")
        print(f"BUILDING: {brief.objective.splitlines()[0]}")
        print(f"{'='*60}")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Resume: {resume}")
        print(f"  Coding agent: {coding_agent or 'from config'}")
        if eval_dir:
            print(f"  Evaluation: {eval_dir}")
        if data_dir:
            print(f"  Data: {data_dir}")

        # Resolve the starting repository: a URL, a local path, or a
        # workflow search over the knowledge graph.
        resolved_repo = (
            None
            if resume
            else self._resolve_initial_repo(initial_repo, brief.objective)
        )
        if resolved_repo:
            print(f"  Repo: {resolved_repo}")
        print()

        problem = self._build_problem_description(goal_text)

        # The harness: the loaded one wins over the brief's line (as call
        # arguments win for budgets); a resume with none loaded takes the
        # launch record's source first (absolute, resolvable from any
        # directory), then the brief's line. A resume compares the hash
        # with the checkpoint's and refuses a changed harness by name.
        harness = self._harness
        if harness is not None:
            # The directory as it is at the call, not at the load: an edit
            # since changes the hash (and a resume compares that hash), a
            # fix since clears a row. The object keeps the fresh handle.
            harness = reload_harness(harness, settings=self._harness_settings)
            self._harness = harness
        if harness is None and record is not None and record.get("harness"):
            harness = load_harness(record["harness"]["source"], settings=self._harness_settings)
        if harness is None and brief.harness:
            harness = load_harness(brief.harness, settings=self._harness_settings)
        if checkpoint is not None:
            recorded = checkpoint.harness
            changed = (recorded is None) != (harness is None) or (
                recorded is not None and recorded["hash"] != harness.hash
            )
            if changed:
                raise RunCheckpointIncompatibleError(resume_refusal(recorded, harness))
        if harness is not None:
            unmet = harness.missing_for(BUILD_MODULES)
            if unmet:
                raise HarnessError(unmet_message(harness, unmet, "build"))
            if coding_agent is not None:
                host_for(coding_agent)   # a host with no staging is refused here, not inside the first session
            # A variable put in .env since this process started reaches the
            # sessions, as the prerequisite check above already counted it.
            load_dotenv(find_dotenv(usecwd=True))
            overrode = self._harness is not None and bool(brief.harness) and not _same_reference(brief.harness, harness)
            print(
                f"  Harness: {harness.label()} from {harness.source}"
                + (f" (loaded; wins over the brief's line {brief.harness!r})" if overrode else "")
            )
            added = harness.sessions_added()
            print(f"  Harness sessions: {'; '.join(added) if added else 'none (pages and attachments only)'}")
            for line in advisory_lines(harness, BUILD_MODULES):
                print(line)

        # The experience: the loaded sources are what is served, and a
        # loaded list wins over the brief's Experience line as the loaded
        # harness wins over its Harness line; a fresh build with nothing
        # loaded takes the brief's items (a resume took the record's list
        # above, with its pins).
        if not self._experience and record is None:
            for reference in brief.experience:
                self.load_experience(reference)

        # Experience serving (docs/plans/redesign/wp3-experience.md): what
        # is loaded is what is served, in two phases. The PLAN (the gate
        # environment, the heads; no filesystem writes) happens before
        # orchestrator construction so the gates it mounts can be
        # fingerprinted into the strategy params; the STAGE (one clone per
        # bank, the intro, the serving record) after, once the workspace
        # exists — a seeded workspace must be EMPTY at construction, so
        # nothing may touch it earlier. Nothing loaded -> nothing served,
        # nothing staged.
        serving_plan = None
        serving = None
        if self._experience:
            output_path = output_path or _absolute(os.path.join(
                "tmp", "search_strategy_workspace", uuid.uuid4().hex
            ))
            serving_plan = plan_campaign_serving(
                self._experience,
                task_coordinates(harness.name if harness is not None else None),
                output_path,
                self._experience_settings["probe_budget"],
            )
            # The search over a loaded index, connected from its record
            # (the plan has refused a second index already); its backend
            # must hold the pages the record counts.
            indexes = [source for source in self._experience if source.kind == "index"]
            if indexes:
                store = ExperienceStore(self._experience_settings["banks"], self._experience_settings["indexes"])
                index_record = store.index_record(indexes[0].reference)
                # The pin holds: the record on disk is the index the
                # handle loaded, not one ingested under the name since.
                if index_record.id != indexes[0].head:
                    raise ExperienceError(
                        f"{indexes[0].reference} was re-ingested since it was loaded (index {indexes[0].head} "
                        f"then, {index_record.id} now); load it again to serve the new index"
                    )
                self._close_knowledge_search()
                self.knowledge_search = search_for(index_record, self._config)
                if not self.knowledge_search.validate_backend_data():
                    raise ExperienceError(
                        f"the index `{index_record.name}` has no pages in its stores (collection "
                        f"{index_record.backend_refs.get('weaviate_collection')} at "
                        f"{index_record.backend_refs.get('weaviate_url')}); the stores were wiped since it was "
                        f"added — `kapso experience remove {index_record.name}` and add it again"
                    )

        # The engine's problem text: the brief, and where the evaluation
        # and the data live.
        handler = GenericProblemHandler(
            problem_description=problem,
            eval_dir=eval_dir,
            data_dir=data_dir,
        )

        # Create orchestrator
        # The status is born inside this construction (the strategy's
        # constructor); whatever raises after that, in the constructor's
        # tail or the orchestrator's own setup, is published on the file,
        # whose writer went with the exception (wp6-observability.md).
        try:
            orchestrator = OrchestratorAgent(
                handler,
                config_path=self.config_path,
                coding_agent=coding_agent,
                knowledge_search=self.knowledge_search if self.knowledge_search.is_enabled() else None,
                # The orchestration layer owns the experiment workspace (a git
                # repo with branches); `output` IS that workspace, so the
                # solution's trace is a real git repo with the record under
                # .kapso/.
                workspace_dir=output_path,
                resume=resume,
                harness=harness,
                experience=[vars(source.ref()) for source in self._experience],
                iteration_evaluator=iteration_evaluator,
                iteration_evaluator_failure_policy=(
                    iteration_evaluator_failure_policy
                ),
                initial_repo=resolved_repo,
                eval_dir=eval_dir,
                data_dir=data_dir,
                goal=goal_text,
                strategy_params_overrides=_memory_overrides(serving_plan),
            )
        finally:
            in_flight = sys.exc_info()[1]
            if in_flight is not None and output_path is not None:
                mark_failed(Path(output_path) / ".kapso" / "status.json", in_flight)
                notify_build_end(self._config["notify"], Path(output_path), brief.objective, launched_at)

        # Stage serving now that the workspace exists (see the plan/stage
        # note above): the clones, the record and the intro file the
        # ideation and implementation prompts read at session time (the
        # problem text is frozen at construction and never was a seam).
        # From the orchestrator's construction to `solve()`, a failure (the
        # serving's staging, the launch record) is published as the
        # campaign's through the status the strategy opened, the exception
        # propagating (wp6-observability.md, change 6).
        try:
            if serving_plan is not None:
                serving = stage_serving(serving_plan, self._experience)
                # Fail loud rather than lie: the intro instructs sessions to
                # call the three bank tools, so the gate that provides them
                # must have resolved. A silently-unmounted gate produced an
                # intro advertising tools no session had (E2E review
                # 2026-08-24, blocker 1).
                mounted = orchestrator.search_strategy.ideation_gates
                if serving_plan["banks"] and "bank" not in mounted:
                    raise RuntimeError(
                        "serving staged but the 'bank' gate is not mounted in "
                        f"ideation gates {mounted} — the intro would advertise "
                        "tools the sessions do not have"
                    )
                for line in serving["lines"]:
                    print(f"  Experience: {line}")
                print(f"  Gates mounted: {mounted}")

            # On a fresh campaign, record the launch so a resume (`kapso build
            # --resume`, `kapso inbox reply`) continues with the launch's own
            # brief and flags without the person retyping anything.
            if not resume:
                workspace_dir = orchestrator.search_strategy.workspace.workspace_dir
                write_launch_record(workspace_dir, {
                    "brief": brief.to_dict(),
                    "harness": harness.ref_dict() if harness is not None else None,
                    "experience": [vars(source.ref()) for source in self._experience],
                    "config_path": self._launch_config_path(),
                    # The launch in the handle's words: the host the flag
                    # set (None: the config's), the budget as resolved.
                    "host": coding_agent,
                    "output_path": str(Path(workspace_dir).resolve()),
                    "experiments": max_iterations,
                    "minutes": time_budget_minutes,
                    "usd": cost_budget,
                    "reserve_minutes": finalization_reserve_minutes,
                    "eval_dir": _absolute(eval_dir),
                    "data_dir": _absolute(data_dir),
                    "resumable_from_inbox": iteration_evaluator is None,
                    "dotenv_path": find_dotenv(usecwd=True),
                })
                # A build that named no output directory is registered now
                # that the orchestrator has made one; every other launch was
                # registered before its checks.
                if not registered:
                    register_campaign(inbox_block["registry"], workspace_dir, brief.objective)
        finally:
            in_flight = sys.exc_info()[1]
            if in_flight is not None:
                # The strategy's status (a test double may have none).
                status = getattr(orchestrator.search_strategy, "operation_status", None)
                if status is not None and not status.terminal:
                    status.failed(in_flight)
                notify_build_end(
                    self._config["notify"], Path(orchestrator.search_strategy.workspace.workspace_dir), brief.objective, launched_at,
                )

        # The status is born, so the launch receipt has told what it had
        # to: a campaign that ran is known by its status from here on, and
        # a deleted one is not listed forever on a leftover receipt.
        launch_receipt_path(Path(orchestrator.search_strategy.workspace.workspace_dir)).unlink(missing_ok=True)

        # The delivery runs inside the orchestrator's protected region,
        # after the search and before the terminal publish: a watcher that
        # sees `done` finds the record, and a checkout that raises is
        # published as the campaign's failure, with no record and a
        # checkpoint a resume can finish from (wp6-observability.md).
        delivered: List[Tuple[Solution, Any]] = []

        def finish(solve_result) -> None:
            solution = _deliver(orchestrator, solve_result, brief=brief, harness=harness, experience=self._experience)
            delivered.append((solution, solve_result))

        # Run experimentation
        print("Running experiments...")
        try:
            orchestrator.solve(
                experiment_max_iter=max_iterations,
                time_budget_minutes=time_budget_minutes,
                cost_budget=cost_budget,
                finalization_reserve_minutes=finalization_reserve_minutes,
                on_status=on_status,
                finish=finish,
            )
        finally:
            notify_build_end(
                self._config["notify"], Path(orchestrator.search_strategy.workspace.workspace_dir), brief.objective, launched_at,
            )
        [(solution, solve_result)] = delivered
        strategy = orchestrator.search_strategy

        print(f"\n{'='*60}")
        print("Build complete")
        print(f"{'='*60}")
        print(solution.explain())
        # Every gate call across the campaign's nodes, per gate and tool,
        # and what the metered gates reported spending.
        campaign_phases = {
            f"{node.node_id}:{phase}": values
            for node in strategy.get_experiment_history()
            for phase, values in node.phase_telemetry.items()
        }
        knowledge = knowledge_summary(campaign_phases)
        if knowledge:
            print(f"Knowledge consulted: {knowledge}")
        credits = credits_summary(campaign_phases)
        if credits:
            print(f"Knowledge credits: {credits}")
        # The campaign's platform spend against its budget, the same
        # renderer as the watch bar, so the two never drift.
        if solve_result.platform_budget is not None:
            print(f"Platform cost: {platform_cost_summary(solve_result.platform_budget)}")

        return solution

    # =========================================================================
    # THE INBOX (docs/research/evolve-hub-design.md v4)
    # =========================================================================

    @staticmethod
    def inbox(campaign: str) -> List[Request]:
        """The campaign's open requests, oldest first — what the person
        must answer before it continues. Empty when nothing waits."""
        requests = load_requests(inbox_path(campaign))
        return sorted(
            (request for request in requests.values() if request.open),
            key=lambda request: request.id,
        )

    @staticmethod
    def inbox_ideas(campaign: str) -> Dict[int, str]:
        """One line per node of the campaign — the `for` row of a request
        — read from the checkpoint. Empty without a checkpoint."""
        store = RunCheckpointStore(campaign)
        if not store.exists():
            return {}
        nodes = store.load().strategy_state.get("node_history") or []
        return {
            int(node["node_id"]): idea_line(str(node.get("solution", "")))
            for node in nodes
        }

    @classmethod
    def reply(
        cls, campaign: str, request_id: int, note: str = "", *,
        allow_secret: bool = False,
    ) -> Optional[Solution]:
        """Answer a request and, when the node that asked has every request
        answered, resume the campaign in this process: the session that
        asked is continued with the reply. Returns the campaign's result,
        or None when it could not be resumed here (another request still
        open, or a campaign started from a script with a callback — resume
        it from the script with resume=True).

        Refuses while a live process holds the campaign, so nothing runs
        twice, and refuses a note shaped like a credential unless
        allow_secret is True (replies are plain text handed to the coder;
        the value belongs in the .env the fix names). An empty note means
        "done"."""
        campaign_dir = str(Path(campaign).resolve())
        if not Path(campaign_dir, ".kapso").is_dir():
            raise FileNotFoundError(
                f"{campaign_dir} is not a campaign directory on this machine"
            )
        status_file = Path(campaign_dir) / ".kapso" / "status.json"
        if status_file.is_file() and OperationStatusView(status_file).alive:
            pid = OperationStatusView(status_file).data.get("pid")
            raise RuntimeError(
                f"{campaign_dir} is running (pid {pid}); wait for it to pause "
                "before replying"
            )
        path = inbox_path(campaign_dir)
        request = record_reply(path, request_id, note, allow_secret=allow_secret)
        print(f"#{request_id} answered.")
        still_open = sorted(
            other.id
            for other in load_requests(path).values()
            if other.node == request.node and other.open
        )
        if still_open:
            print(
                f"{', '.join(f'#{i}' for i in still_open)} still open, so node "
                f"{request.node} waits; nothing else to run."
            )
            return None
        record = read_launch_record(campaign_dir)
        if record is None or not record.get("resumable_from_inbox"):
            print(
                f"Recorded. This campaign cannot be resumed from the inbox "
                "(started from a script with a callback, or before the inbox "
                "existed) — resume it from your script with resume=True."
            )
            return None
        checkpoint = RunCheckpointStore(campaign_dir).load()
        remaining = max(
            1, int(record["experiments"]) - int(checkpoint.completed_iterations)
        )
        print(
            f"Resuming {campaign_dir}: continuing node {request.node}'s session. "
            "Ctrl-C stops it; resume later with kapso build --output "
            f"{campaign_dir} --resume"
        )
        # Everything else a resume needs comes from the launch record,
        # its .env included: a reply may run far from the launch directory.
        if record.get("dotenv_path"):
            load_dotenv(record["dotenv_path"])
        # Your bank where the launch had it (`Kapso(bank=)`): the record's
        # `mine` entry names its home, absolute.
        mine = next((entry for entry in record.get("experience", []) if entry["reference"] == MINE), None)
        kapso = cls(config=record["config_path"], bank=mine["resolved"] if mine is not None else None)
        return kapso.build(
            output=campaign_dir,
            budget=Budget(experiments=remaining),
            resume=True,
        )

    @staticmethod
    def _validate_resume_workspace(output_path: Optional[str]) -> None:
        """Reject resume requests before they can initialize or mutate a repo."""
        if not output_path:
            raise ValueError("resume=True requires an existing output_path")

        path = Path(output_path).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(
                f"Resume workspace does not exist or is not a directory: {path}"
            )

        if not (path / ".git").is_dir():
            raise ValueError(
                f"Resume workspace is not a Git repository: {path}"
            )
        if git.Repo(str(path), search_parent_directories=False).bare:
            raise ValueError(
                f"Resume workspace must be a non-bare Git repository: {path}"
            )
    
    def deploy(
        self,
        solution: Solution,
        target: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        options: Optional[Dict[str, str]] = None,
        coding_agent: Optional[str] = None,
    ) -> Software:
        """Deploy a solution as running software (docs/plans/redesign/wp5-deploy.md).

        One session on a copy of the solution adapts it to the target,
        deploys it with the target's scripts and proves one call; the
        engine checks the result, records the deployment under the
        campaign (`.kapso/deployments/<n>/`) and hands back the handle.

        Args:
            solution: The Solution from build(), or Solution.open(path); the
                campaign must be checked out at the delivered candidate, its
                tracked files unchanged
            target: Where to deploy: `auto` (the session chooses), a bundled
                target (`local`, `docker`, `modal`, `bentoml`, `langgraph`)
                or a target the build's harness ships; the config's
                `deployment.target` when None
            env: Variables the software needs at runtime, written to `.env`
                in the workspace, named (never valued) in the prompt, set on
                every runtime the engine and the target's scripts manage
            options: Options for the target's workflow (a harness target's
                `schedule`, say), named in the prompt and exported to the
                scripts as KAPSO_OPTION_<NAME>
            coding_agent: The host the deployment session runs on; the
                config's `deployment.cli`, else the build's host

        Returns:
            Software: run(inputs), stop(), start(), is_healthy(), logs(),
            the record's facts; `Software.open(campaign, number)` from any
            later process

        Example:
            solution = kapso.build("Create a trading bot")
            software = kapso.deploy(solution, target="docker")
            result = software.run({"ticker": "AAPL"})
            software.stop()
        """
        if solution.commit is None:
            raise DeployError(
                f"nothing to deploy: the build in {solution.campaign} delivered no "
                f"candidate (stopped: {solution.stopped_reason})"
            )
        checked_out = _git_rev(solution.campaign, "HEAD")
        if checked_out != solution.commit:
            raise DeployError(
                f"{solution.campaign} is checked out at {checked_out[:12]}, not at "
                f"the delivered candidate {solution.commit[:12]} ({solution.branch}); "
                f"run `git -C {solution.campaign} checkout {solution.branch}` first"
            )
        # The harness: the loaded one (the build's name, any version), else
        # the record's; its deployment module rides into the session.
        harness = self._harness_for(solution, verb="deploy")
        chosen, _offered = resolve_target(target, self._config["deployment"]["target"], harness)
        run_preflight(
            "deploy", self._config, target=AUTO if chosen is None else chosen.name, coding_agent=coding_agent,
        )
        print(f"\n{'='*60}")
        print(f"DEPLOYING: {solution.brief.objective.splitlines()[0]}")
        print(f"{'='*60}")
        print(f"  Target: {AUTO if chosen is None else chosen.name}")
        print(f"  Campaign: {solution.campaign}")
        print()
        return run_deployment(
            solution, config=self._config, target=target, env=env, options=options,
            coding_agent=coding_agent, harness=harness,
        )

    def _harness_for(self, solution: Solution, *, verb: str) -> Optional[Harness]:
        """The harness a verb on a finished build runs with: the loaded one
        when its name is the record's (another version is allowed and
        printed beside the build's), else the record's source resolved
        again, else none. A verb refuses while a requirement of its own
        module is unmet."""
        recorded = solution.harness
        loaded = self._harness
        if loaded is not None:
            loaded = reload_harness(loaded, settings=self._harness_settings)
            self._harness = loaded
            if recorded is None and verb == "learn":
                # A learn may run under a harness the build did not have:
                # its learning module shapes the crews' sessions, and the
                # lessons stay scoped to the build's own coordinate (none).
                print(f"  Harness: {loaded.label()} for the learn; the build ran without one")
                return self._ready_for(loaded, verb)
            if recorded is None:
                raise HarnessError(
                    f"{solution.campaign} was built without a harness, and a {verb} runs as its build "
                    f"did: the loaded {loaded.label()} is not attached to a finished campaign"
                )
            if loaded.name != recorded.name:
                raise HarnessError(
                    f"{solution.campaign} was built with harness {recorded.name} {recorded.version}; "
                    f"the loaded harness is {loaded.name} {loaded.version}: a {verb} runs with the "
                    "build's harness (another version of it is allowed)"
                )
            harness = loaded
        elif recorded is not None:
            if not _source_is_reachable(recorded.source):
                raise HarnessError(
                    f"{solution.campaign} was built with harness {recorded.name} {recorded.version} "
                    f"from {recorded.source}, which is not here; pass --harness <ref> (or load one) "
                    f"pointing at a copy of {recorded.name} {recorded.version}"
                )
            harness = load_harness(recorded.source, settings=self._harness_settings)
            if harness.name != recorded.name:
                raise HarnessError(
                    f"{recorded.source} now holds harness {harness.name}, and {solution.campaign} was "
                    f"built with {recorded.name} {recorded.version}; pass --harness <ref> pointing at "
                    f"a copy of {recorded.name}"
                )
        else:
            return None
        if harness.hash != recorded.hash:
            print(
                f"  Harness: {harness.label()} for the {verb}; the build ran with "
                f"{recorded.name} {recorded.version} ({recorded.hash[:12]})"
            )
        return self._ready_for(harness, verb)

    def _ready_for(self, harness: Harness, verb: str) -> Harness:
        """The harness a verb runs with, its own module's requirements met
        and its advisories printed."""
        modules = DEPLOY_MODULES if verb == "deploy" else LEARN_MODULES
        unmet = harness.missing_for(modules)
        if unmet:
            raise HarnessError(unmet_message(harness, unmet, verb))
        for line in advisory_lines(harness, modules):
            print(line)
        return harness
    
    # =========================================================================
    # INITIAL REPO RESOLUTION HELPERS
    # =========================================================================
    
    def _resolve_initial_repo(self, initial_repo: Optional[str], goal: str) -> Optional[str]:
        """
        Resolve initial_repo to a local path.
        
        Handles three cases:
        1. GitHub URL: Clone to temp directory
        2. Local path: Use as-is
        3. None: Search for workflow repo in KG
        
        Args:
            initial_repo: Local path, GitHub URL, or None
            goal: The goal (used for workflow search if initial_repo is None)
            
        Returns:
            Local path to repo, or None if no repo found/provided
        """
        if initial_repo is not None:
            # Check if it's a GitHub URL
            if self._is_github_url(initial_repo):
                return self._clone_github_repo(initial_repo)
            # Assume local path
            return initial_repo
        
        # No initial_repo provided - search for workflow repo
        return self._search_workflow_repo(goal)
    
    def _is_github_url(self, path: str) -> bool:
        """Check if path is a GitHub URL."""
        return (
            path.startswith("https://github.com/") or 
            path.startswith("git@github.com:") or
            path.startswith("http://github.com/")
        )
    
    def _clone_github_repo(self, url: str) -> str:
        """
        Clone a GitHub repository to a temporary directory.
        
        Args:
            url: GitHub repository URL
            
        Returns:
            Local path to cloned repository
        """
        temp_dir = tempfile.mkdtemp(prefix="kapso_repo_")
        print(f"  Cloning {url}...")
        # A clone that fails propagates: a build must not start from an
        # empty repository when the person named one.
        git.Repo.clone_from(url, temp_dir)
        print(f"  Cloned to: {temp_dir}")
        return temp_dir
    
    def _search_workflow_repo(self, goal: str) -> Optional[str]:
        """
        Search for a relevant workflow repository in the Knowledge Graph.
        
        Args:
            goal: The goal to search for
            
        Returns:
            Local path to cloned workflow repo, or None if not found
        """
        # Only search if KG is enabled
        if not self.knowledge_search.is_enabled():
            print("  No KG index - skipping workflow search")
            return None
        
        print("  Searching for relevant workflow...")
        workflow_search = WorkflowRepoSearch(kg_search=self.knowledge_search)
        result = workflow_search.search(goal, top_k=1)
        if not result.is_empty and result.top_result.github_url:
            starter_url = result.top_result.github_url
            print(f"  Found workflow repo: {starter_url}")
            return self._clone_github_repo(starter_url)
        print("  No matching workflow found")
        return None
    
    def _build_problem_description(self, goal: str) -> str:
        """Build the full problem description for the orchestrator."""
        return f"# Goal\n{goal}"
    


# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

__all__ = [
    "Kapso",
    "Brief",
    "Budget",
    "Solution",
    "Software",
    "DeployError",
    "IterationEvaluator",
]
