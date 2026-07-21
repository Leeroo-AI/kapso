"""Real Git/filesystem fixtures for the M3 capture pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from kapso.core.config import load_config
from kapso.cross_run.capture.exporter import RunCaptureRequest
from kapso.cross_run.contracts import (
    ArtifactEnvironment,
    CompletionState,
    EvaluationFingerprint,
    ExpertScopeContract,
    LaunchManifest,
    TaskContextBinding,
)
from kapso.cross_run.settings import CrossRunSettings
from kapso.execution.fidelity import EvaluationAttempt
from kapso.execution.memories.experiment_memory import ExperimentHistoryStore
from kapso.execution.run_checkpoint import RunCheckpoint, RunCheckpointStore
from test_cross_run_contracts import build_records, digest
from test_ideation_resume import finalized_node, linked_strategy

RUN_ID = "run_" + "a" * 32
STARTED_AT = "2026-07-20T12:00:00Z"


@dataclass
class CaptureFixture:
    workspace: Path
    request: RunCaptureRequest
    settings: CrossRunSettings
    strategy: object
    store: ExperimentHistoryStore
    checkpoint_store: RunCheckpointStore

    def save_checkpoint(self, status: str) -> None:
        checkpoint = RunCheckpoint.create(
            strategy_type="generic",
            goal="Improve the complete task solution.",
            config_fingerprint=digest("run-config"),
            status=status,
            completed_iterations=1,
            cumulative_cost=0.25,
            current_feedback=self.strategy.node_history[0].feedback,
            strategy_state=self.strategy.dump_state(),
            elapsed_seconds=2.0,
            cost_by_component={"implementation": 0.25},
            last_stop=None,
        )
        self.checkpoint_store.save(checkpoint)


def make_capture_fixture(
    tmp_path: Path,
    *,
    completion_state: CompletionState = CompletionState.STOPPED,
    secret_source: bool = False,
    forbidden_artifacts: bool = False,
    unapproved_license: bool = False,
    multiple_unapproved_licenses: bool = False,
    denied_name_variants: bool = False,
    excluded_artifact_classes: bool = False,
    raw_observation_sentinel: str | None = None,
) -> CaptureFixture:
    settings = CrossRunSettings.from_dict(
        load_config("src/kapso/config.yaml")["cross_run"]
    )
    strategy, archive, repo = linked_strategy(tmp_path)
    baseline_commit = repo.head.commit.hexsha
    repo.git.checkout("generic_exp_0")
    if secret_source:
        source = "API_KEY = 'super-secret-production-token'\n"
    elif multiple_unapproved_licenses:
        source = (
            "# SPDX-License-Identifier: GPL-3.0-only\n"
            "# SPDX-License-Identifier: AGPL-3.0-only\n"
            "VALUE = 1\n"
        )
    elif unapproved_license:
        source = "# SPDX-License-Identifier: GPL-3.0-only\nVALUE = 1\n"
    else:
        source = "VALUE = 1\n"
    (Path(strategy.workspace_dir) / "solution.py").write_text(source, encoding="utf-8")
    if forbidden_artifacts:
        (Path(strategy.workspace_dir) / "weights.pt").write_bytes(b"model weights")
        (Path(strategy.workspace_dir) / "training.csv").write_text(
            "label,value\n1,2\n", encoding="utf-8"
        )
        evaluation_dir = Path(strategy.workspace_dir) / "kapso_evaluation"
        evaluation_dir.mkdir()
        (evaluation_dir / "evaluate.py").write_text(
            "print('hidden evaluator')\n", encoding="utf-8"
        )
    if denied_name_variants:
        (Path(strategy.workspace_dir) / "credentials.json").write_text(
            "{}\n", encoding="utf-8"
        )
        (Path(strategy.workspace_dir) / ".gitconfig").write_text(
            "[safe]\n", encoding="utf-8"
        )
        (Path(strategy.workspace_dir) / ".git-credentials").write_text(
            "none\n", encoding="utf-8"
        )
        nested = Path(strategy.workspace_dir) / "nested"
        nested.mkdir()
        (nested / "prod_credentials.toml").write_text(
            "enabled = false\n", encoding="utf-8"
        )
        denied_credential_paths = (
            ".netrc",
            ".npmrc",
            ".pypirc",
            ".aws/config.json",
            ".azure/settings.json",
            ".docker/config.json",
            ".gnupg/settings.json",
            ".ssh/config.json",
            "prod_credential.json",
            "secret_key.py",
        )
        for relative_path in denied_credential_paths:
            credential_path = Path(strategy.workspace_dir) / relative_path
            credential_path.parent.mkdir(parents=True, exist_ok=True)
            credential_path.write_text("{}\n", encoding="utf-8")
    artifact_class_paths = (
        ".svn/entries.json",
        ".cache/preprocessed.json",
        "__pycache__/metadata.json",
        "data/train.json",
        "training_data/samples.jsonl",
        "hidden_evaluator/test_cases.json",
        "evaluation/private.yaml",
        "weights/model.json",
    )
    legitimate_source_paths = (
        "cache_adapter.py",
        "data.py",
        "dataset_reader.py",
        "evaluation_utils.py",
    )
    if excluded_artifact_classes:
        for relative_path in artifact_class_paths:
            artifact_path = Path(strategy.workspace_dir) / relative_path
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text("{}\n", encoding="utf-8")
        for relative_path in legitimate_source_paths:
            (Path(strategy.workspace_dir) / relative_path).write_text(
                "VALUE = 1\n", encoding="utf-8"
            )
    committed_paths = ["solution.py"]
    if forbidden_artifacts:
        committed_paths.extend(
            ("weights.pt", "training.csv", "kapso_evaluation/evaluate.py")
        )
    if denied_name_variants:
        committed_paths.extend(
            (
                "credentials.json",
                ".gitconfig",
                ".git-credentials",
                "nested/prod_credentials.toml",
                *denied_credential_paths,
            )
        )
    if excluded_artifact_classes:
        committed_paths.extend(artifact_class_paths)
        committed_paths.extend(legitimate_source_paths)
    repo.git.add(committed_paths)
    repo.git.commit("-m", "candidate")
    candidate_commit = repo.head.commit.hexsha
    repo.git.checkout("main")

    node = finalized_node()
    node.workspace_dir = strategy.workspace_dir
    node.implementation_base_ref = baseline_commit
    node.diff_base_ref = baseline_commit
    node.feedback_base_ref = baseline_commit
    records = build_records()
    evaluation = next(
        item for item in records if isinstance(item, EvaluationFingerprint)
    )
    evaluator_id = evaluation.evaluator_fingerprint.removeprefix("sha256:")
    node.evaluation_attempts = [
        EvaluationAttempt(
            commit_sha=candidate_commit,
            evaluator_id=evaluator_id,
            fidelity="full",
            fraction=1.0,
            seed=1,
            score=node.score,
            duration_seconds=1.0,
            metrics={"quality": node.score},
        )
    ]
    node.metrics = {"quality": node.score}
    node.primary_metric = "quality"
    if raw_observation_sentinel is not None:
        node.agent_output = raw_observation_sentinel
        node.code_changes_summary = raw_observation_sentinel
        node.code_diff = raw_observation_sentinel
        node.evaluation_output = raw_observation_sentinel
        node.evaluation_script_path = raw_observation_sentinel
        node.feedback = raw_observation_sentinel
        node.technical_difficulties = raw_observation_sentinel
        node.external_evaluation_metadata = {
            "raw_observation": raw_observation_sentinel
        }
        node.metrics[raw_observation_sentinel] = 0.5
        node.phase_telemetry = {
            raw_observation_sentinel: {raw_observation_sentinel: 0.5}
        }
        node.evaluation_attempts[0] = replace(
            node.evaluation_attempts[0],
            metrics={
                **node.evaluation_attempts[0].metrics,
                raw_observation_sentinel: 0.5,
            },
        )
    strategy.node_history = [node]
    strategy.iteration_count = 1
    strategy.previous_errors = []
    strategy.scores_evaluator_id = evaluator_id
    strategy.evaluator_transition = None

    workspace = Path(strategy.workspace_dir)
    history_path = workspace / ".kapso" / "experiment_history.json"
    journal_path = (
        workspace / settings.capture.state_path / settings.capture.journal_filename
    )
    store = ExperimentHistoryStore(
        str(history_path),
        objective_direction="maximize",
        require_idea_links=True,
        run_id=RUN_ID,
        campaign_id=strategy.ideation_campaign_id,
        journal_path=str(journal_path),
        git_command_timeout_seconds=settings.capture.git_command_timeout_seconds,
        git_command_output_bytes=settings.capture.git_command_output_bytes,
    )
    store.add_experiment(node)
    strategy.record_finalized_idea_outcome(node)

    checkpoint_store = RunCheckpointStore(
        str(workspace),
        settings.capture.checkpoint_path,
    )
    scope = next(item for item in records if isinstance(item, ExpertScopeContract))
    context = next(item for item in records if isinstance(item, TaskContextBinding))
    environment = next(
        item for item in records if isinstance(item, ArtifactEnvironment)
    )
    launch = next(item for item in records if isinstance(item, LaunchManifest))
    request = RunCaptureRequest(
        workspace_dir=workspace,
        idea_archive_path=archive.path,
        scope_contract_id=scope.scope_contract_id,
        scope_id=scope.scope_id,
        run_id=RUN_ID,
        campaign_id=strategy.ideation_campaign_id,
        configuration_fingerprint=settings.configuration_fingerprint,
        completion_state=completion_state,
        started_at=STARTED_AT,
        kapso_commit="0" * 40,
        launch_manifest_id=launch.launch_manifest_id,
        knowledge_snapshot_id=launch.knowledge_snapshot_id,
        expert_base_release_id=environment.expert_base_release_id,
        task_context_binding=context,
        artifact_environment=environment,
        evaluation_fingerprints=(evaluation,),
    )
    fixture = CaptureFixture(
        workspace=workspace,
        request=request,
        settings=settings,
        strategy=strategy,
        store=store,
        checkpoint_store=checkpoint_store,
    )
    fixture.save_checkpoint(
        "completed" if completion_state is CompletionState.COMPLETE else "running"
    )
    return fixture
