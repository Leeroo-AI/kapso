"""End-to-end contract test for the canonical Generic ideation lifecycle."""

import copy
import hashlib
import json
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import git

from kapso.execution.budget import BudgetSnapshot
from kapso.core.config import load_config
from kapso.execution.evaluation_integrity import AGENT_GENERATED
from kapso.execution.fidelity import FULL_PASSTHROUGH
from kapso.execution.memories.experiment_memory import ExperimentHistoryStore
from kapso.execution.orchestrator import OrchestratorAgent
from kapso.execution.search_strategies.generic.ideation import (
    AnalyzerSettings,
    BatchStatus,
    CampaignEvidenceBuilder,
    CandidateAnalyzer,
    CandidateGenerator,
    CandidateGeneratorSettings,
    CandidateSelector,
    EmbeddingBatch,
    EmbeddingRecord,
    EmbeddingSettings,
    EmbeddingTelemetry,
    EvidenceAuthor,
    EvidenceSettings,
    GapPrioritySettings,
    GenerationMemberSettings,
    IdeaArchive,
    IdeaStatus,
    IdeationEngine,
    OperatorSettings,
    new_identifier,
)
from kapso.execution.search_strategies.generic.strategy import GenericSearch
from test_ideation_engine import PacketRunner


def _canonical_ideation_config() -> dict:
    config_path = Path(__file__).parents[1] / "src" / "kapso" / "config.yaml"
    config = load_config(str(config_path))
    return copy.deepcopy(config["ideation_profiles"]["DEFAULT"])


class DeterministicEmbeddingProvider:
    """Exercise the embedding contract without making a network call."""

    def __init__(self, settings: EmbeddingSettings):
        self.settings = settings
        self.call_count = 0

    def embed(self, texts):
        inputs = tuple(texts)
        self.call_count += 1
        records = []
        for index, value in enumerate(inputs):
            vector = [0.0] * self.settings.dimensions
            vector[index % self.settings.dimensions] = 1.0
            records.append(
                EmbeddingRecord(
                    provider="openai",
                    model=self.settings.model,
                    dimensions=self.settings.dimensions,
                    input_hash=hashlib.sha256(value.encode("utf-8")).hexdigest(),
                    vector=tuple(vector),
                )
            )
        return EmbeddingBatch(
            records=tuple(records),
            telemetry=EmbeddingTelemetry(
                provider="openai",
                model=self.settings.model,
                call_count=1,
                input_tokens=len(inputs),
                duration_seconds=0.01,
            ),
        )


def _git_workspace(path: Path) -> git.Repo:
    path.mkdir(parents=True)
    repo = git.Repo.init(path)
    with repo.config_writer() as config:
        config.set_value("user", "name", "Ideation Integration Test")
        config.set_value("user", "email", "ideation-integration@example.com")
    path.joinpath("solution.txt").write_text("baseline\n", encoding="utf-8")
    repo.git.add("solution.txt")
    repo.index.commit("baseline")
    repo.git.branch("-M", "main")
    return repo


def _strategy(path: Path, repo: git.Repo, ideation_config: dict) -> GenericSearch:
    strategy = GenericSearch.__new__(GenericSearch)
    strategy.workspace_dir = str(path)
    strategy.workspace = SimpleNamespace(repo=repo)
    strategy.problem_handler = SimpleNamespace(maximize_scoring=True)
    strategy.ideation_config = copy.deepcopy(ideation_config)
    strategy.ideation_campaign_id = new_identifier("campaign")
    strategy.idea_archive = None
    strategy.active_batch_id = None
    strategy.iteration_count = 0
    strategy.node_history = []
    strategy.previous_errors = []
    strategy.scores_evaluator_id = ""
    strategy.evaluator_transition = None
    strategy.feedback_generator = None
    strategy.goal = "Improve the complete task solution."
    strategy.evaluation_provenance = AGENT_GENERATED
    strategy.provided_evaluation_manifest = {}
    strategy.provided_evaluation_fingerprint = None
    strategy.registered_evaluation_manifest = {}
    strategy.registered_evaluation_command = (
        "python kapso_evaluation/kapso_eval.py "
        "--fidelity full --fraction 1.0 --seed 17"
    )
    strategy.registered_evaluator_id = "evaluator-integration-v1"
    strategy.registered_subsample_seed = 17
    strategy.registered_data_manifest = {}
    strategy.fidelity_decision = FULL_PASSTHROUGH
    strategy.budget_snapshot = BudgetSnapshot(
        iteration_index=0,
        max_iterations=3,
        elapsed_seconds=0.0,
        cost_usd=0.0,
        time_budget_seconds=600.0,
        finalization_reserve_seconds=60.0,
    )
    strategy.budget_snapshot_monotonic = time.monotonic()

    @contextmanager
    def materialize_parent(_parent):
        yield str(path)

    strategy._materialize_ideation_parent = materialize_parent
    strategy.workspace.materialize_ref = materialize_parent
    return strategy


def _engine(
    strategy: GenericSearch,
    runner: PacketRunner,
    embedding_provider: DeterministicEmbeddingProvider,
) -> IdeationEngine:
    coding_agents = strategy.ideation_config["coding_agents"]
    evidence = dict(strategy.ideation_config["evidence"])
    evidence["evaluator_id"] = strategy.registered_evaluator_id
    evidence["comparable_seed"] = strategy.registered_subsample_seed
    return IdeationEngine(
        archive=strategy._ensure_idea_archive(),
        evidence_builder=CampaignEvidenceBuilder(EvidenceSettings.from_dict(evidence)),
        operator_settings=OperatorSettings.from_dict(
            strategy.ideation_config["operators"]
        ),
        gap_priority_settings=GapPrioritySettings.from_dict(
            strategy.ideation_config["gaps"]
        ),
        generator=CandidateGenerator(
            runner,
            CandidateGeneratorSettings.from_dict(coding_agents["generator"]),
        ),
        analyzer=CandidateAnalyzer(
            AnalyzerSettings.from_dict(strategy.ideation_config["analyzer"]),
            embedding_provider=embedding_provider,
        ),
        selector=CandidateSelector(
            runner,
            GenerationMemberSettings.from_dict(coding_agents["selector"]),
        ),
    )


def _install_execution_boundary(strategy: GenericSearch, repo: git.Repo) -> None:
    def implement(*, solution, problem, branch_name, parent_branch_name):
        del solution, problem
        repo.create_head(branch_name, repo.commit(parent_branch_name))
        repo.git.checkout(branch_name)
        Path(strategy.workspace_dir, "solution.txt").write_text(
            "implemented candidate\n",
            encoding="utf-8",
        )
        repo.git.add("solution.txt")
        repo.index.commit("implement selected idea")
        repo.git.checkout("main")
        strategy._last_implementation_success = True
        strategy._last_implementation_error = ""
        return (
            "<code_changes_summary>Implemented one intervention.</code_changes_summary>\n"
            "<evaluation_script_path>kapso_evaluation/evaluate.py</evaluation_script_path>\n"
            "<evaluation_output>KAPSO_EVAL_MANIFEST "
            '{"fidelity":"full","fraction":1.0,"seed":17,'
            '"items":40,"total_items":40,"score":0.75}'
            "</evaluation_output>\n"
            "<score>0.75</score>\n"
            "<technical_difficulties>No technical difficulty.</technical_difficulties>",
            {"cost_usd": 0.2, "duration_seconds": 1.0},
        )

    def generate_feedback(node):
        node.feedback = "The selected intervention improved the canonical score."
        node.evaluation_valid = True
        node.score = 0.75
        return node

    strategy._implement = implement
    strategy.enforce_evaluation_integrity = lambda _node: True
    strategy._generate_feedback = generate_feedback


def test_generic_ideation_archive_memory_checkpoint_and_resume_are_one_lifecycle(
    tmp_path,
):
    workspace = tmp_path / "workspace"
    repo = _git_workspace(workspace)
    ideation_config = _canonical_ideation_config()
    strategy = _strategy(workspace, repo, ideation_config)
    runner = PacketRunner(tmp_path)
    embedding_provider = DeterministicEmbeddingProvider(
        EmbeddingSettings.from_dict(ideation_config["embeddings"])
    )
    strategy._build_ideation_engine = lambda: _engine(
        strategy,
        runner,
        embedding_provider,
    )
    strategy._build_ideation_evidence_author = lambda: EvidenceAuthor(
        runner,
        GenerationMemberSettings.from_dict(
            ideation_config["coding_agents"]["evidence_author"]
        ),
    )
    _install_execution_boundary(strategy, repo)

    node = strategy.run("Improve the complete task solution.")

    archive = strategy._ensure_idea_archive()
    batch = archive.get_batch(node.selection_batch_id)
    selected = archive.get_idea(node.idea_id)
    assert Counter(runner.roles) == Counter(
        {
            "candidate_0": 1,
            "candidate_1": 1,
            "candidate_selector": 1,
            "evidence_author": 1,
        }
    )
    assert runner.roles[-1] == "evidence_author"
    assert embedding_provider.call_count == 1
    assert batch.status == BatchStatus.BRIDGED
    assert selected.status == IdeaStatus.IMPLEMENTING
    assert selected.experiment_node_id == node.node_id == 0
    assert node.implementation_base_ref == repo.commit("main").hexsha
    assert node.evaluation_attempts[0].score == 0.75
    assert node.phase_telemetry["ideation"]["coding_agent_call_count"] == 3.0
    assert node.phase_telemetry["implementation"]["cost_usd"] == 0.2
    assert node.phase_telemetry["evidence_author"]["coding_agent_call_count"] == 1.0
    assert len(node.external_evaluation_metadata["ideation_evidence"]["claims"]) == 1
    assert node.cost_usd == 0.2

    # Simulate the crash window where durable execution and outcome advance
    # after the last run checkpoint was written.
    stale_checkpoint = strategy.dump_state()
    history = ExperimentHistoryStore(
        str(workspace / ".kapso" / "experiment_history.json"),
        objective_direction="maximize",
        require_idea_links=True,
    )
    orchestrator = SimpleNamespace(
        experiment_store=history,
        search_strategy=strategy,
    )
    OrchestratorAgent._persist_finalized_candidates(orchestrator, [node])
    record = history.experiments[0]

    persisted = IdeaArchive(
        workspace / ideation_config["archive_path"],
        strategy.ideation_campaign_id,
    )
    completed_batch = persisted.get_batch(node.selection_batch_id)
    completed_idea = persisted.get_idea(node.idea_id)
    assert completed_batch.status == BatchStatus.COMPLETED
    assert completed_idea.status == IdeaStatus.EVALUATED
    assert completed_idea.outcome.normalized_delta == 0.75
    assert completed_idea.outcome.supported_claim_ids == (
        persisted.state.claims[0].claim_id,
    )
    assert all(idea.embedding is not None for idea in persisted.state.ideas)
    assert record.idea_id == completed_idea.idea_id
    assert persisted.revision > stale_checkpoint["archive_revision"]

    history = ExperimentHistoryStore(
        str(workspace / ".kapso" / "experiment_history.json"),
        objective_direction="maximize",
        require_idea_links=True,
    )
    resumed = _strategy(workspace, repo, ideation_config)
    resumed.load_state(stale_checkpoint)
    revision_before_reconcile = resumed._ensure_idea_archive().revision
    resumed.reconcile_experiment_memory(history)
    resumed.reconcile_experiment_memory(history)

    assert resumed.active_batch_id is None
    assert len(resumed.node_history) == 1
    assert len(history.experiments) == 1
    assert len(resumed._ensure_idea_archive().state.batches) == 1
    assert len(resumed._ensure_idea_archive().state.ideas) == 2
    assert resumed.node_history[0].idea_id == completed_idea.idea_id
    assert history.experiments[0].selection_batch_id == completed_batch.batch_id
    assert resumed.node_history[0].phase_telemetry == node.phase_telemetry
    assert runner.roles[-1] == "evidence_author"
    assert embedding_provider.call_count == 1
    assert resumed.dump_state()["archive_revision"] == revision_before_reconcile
    assert revision_before_reconcile == persisted.revision
    json.loads(
        (workspace / ".kapso" / "experiment_history.json").read_text(encoding="utf-8")
    )
