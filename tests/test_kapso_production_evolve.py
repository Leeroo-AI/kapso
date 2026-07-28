from __future__ import annotations

from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.launch.contracts import LaunchTaskContextRequest
from kapso.cross_run.launch.production_evolution import ProductionEvolutionResult
from kapso.kapso import Kapso

_CONFIG_PATH = "src/kapso/config.yaml"


def _task_context():
    fingerprint = tree_or_blob_digest(b"public evolve task")
    return LaunchTaskContextRequest.mint(
        capability_tags=("predict",),
        input_contract_fingerprint=fingerprint,
        target_contract_fingerprint=fingerprint,
        starting_artifact_refs=("dataset",),
        method_fingerprint=fingerprint,
        toolchain_fingerprint=fingerprint,
        dependency_runtime_fingerprint=fingerprint,
        budget_hardware_envelope={"hardware": "cpu"},
        transfer_dimensions={},
    )


def test_kapso_evolve_uses_only_the_production_composition(
    tmp_path,
    monkeypatch,
):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "rows.json").write_text("[]", encoding="utf-8")
    captured = {}

    def execute(**arguments):
        captured.update(arguments)
        return ProductionEvolutionResult(
            code_path=(tmp_path / "run" / "workspace").absolute(),
            metadata={
                "launch_manifest_id": "launch-manifest:sha256:" + "1" * 64,
                "expert_release_id": "expert-base-release:sha256:" + "2" * 64,
                "knowledge_snapshot_id": "knowledge-snapshot:sha256:" + "3" * 64,
                "task_adapter_manifest_id": (
                    "task-adapter-manifest:sha256:" + "4" * 64
                ),
                "action_result": {
                    "implementation_summary": "Implemented one successor."
                },
            },
        )

    monkeypatch.setattr("kapso.kapso.execute_production_evolution", execute)

    result = Kapso(config_path=_CONFIG_PATH).evolve(
        goal="Improve the public fixture.",
        output_path=str(tmp_path / "run"),
        task_context_request=_task_context(),
        starting_artifact_sources={"dataset": (dataset, "inputs/dataset")},
        dependency_runtime_contract={"runtime": "python"},
        budget_fidelity_envelope={"fidelity": "full"},
        config_path=_CONFIG_PATH,
        scope_id="ml_ai",
        task_family_id="relational_tabular_prediction",
        task_adapter_id="relbench",
        coding_agent="codex",
    )

    assert result.experiment_logs == ["Implemented one successor."]
    assert result.metadata["launch_manifest_id"].startswith("launch-manifest:")
    assert captured["effective_config"].mode_name == "GENERIC"
    assert captured["run_root"] == (tmp_path / "run").absolute()
    assert captured["state_root"] == tmp_path.absolute()
    assert captured["starting_artifact_sources"] == {
        "dataset": (dataset.absolute(), "inputs/dataset")
    }
    assert captured["task_context_request"] == _task_context()
