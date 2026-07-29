"""The production driver checkpoints one canonical cross-stage receipt."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace

import pytest

import kapso.cross_run.production_smoke as smoke_module
from kapso.core.config import load_effective_config
from kapso.core.embedding_contracts import (
    EmbeddingBatch,
    EmbeddingRecord,
    EmbeddingTelemetry,
    complete_input_hash,
)
from kapso.cross_run.canonical import (
    canonical_json_bytes,
    content_id,
    parse_json_bytes,
    tree_or_blob_digest,
)
from kapso.cross_run.capture.bundle import RunBundleStore
from kapso.cross_run.catalog.projector import RunBundleProjector
from kapso.cross_run.catalog.service import CrossRunCatalog
from kapso.cross_run.contracts import ExpertBaseReleaseManifest
from kapso.cross_run.expert.triggers import (
    ExpertSourceBaseTreeReceipt,
    ExpertTriggerEvidencePacket,
    ExpertTriggerEvaluator,
    ExpertTriggerObservation,
    ExpertTriggerObservationKind,
)
from kapso.cross_run.github.materializer import SourceArchiveExtractionReceipt
from kapso.cross_run.github.command import GitHubCompareAndSwapError
from kapso.cross_run.knowledge.publisher import KnowledgeSnapshotPublisher
from kapso.cross_run.production_smoke import (
    ProductionSmokeError,
    run_production_smoke,
)
from test_knowledge_snapshot_publisher import (
    DeterministicEmbeddingProvider,
    RecordingPublicationAuthority,
)
from test_cross_run_retrieval import snapshot_and_index, source_fixture
from test_expert_triggers import inspection_operation, trigger_packet, trigger_settings

_CONFIG_PATH = "src/kapso/config.yaml"
_EXPERT_RELEASE_ID = "expert-base-release:sha256:" + "1" * 64
_TASK_ADAPTER_MANIFEST_ID = "task-adapter-manifest:sha256:" + "2" * 64
_TASK_ADAPTER_VERIFICATION_RECEIPT_ID = (
    "task-adapter-verification-receipt:sha256:" + "3" * 64
)


def _embedding_batch(settings, texts, vectors):
    return EmbeddingBatch(
        records=tuple(
            EmbeddingRecord(
                provider=settings.provider,
                model=settings.model,
                dimensions=settings.dimensions,
                canonicalizer_version=settings.canonicalizer_version,
                input_hash=complete_input_hash(value),
                vector=vector,
            )
            for value, vector in zip(texts, vectors)
        ),
        telemetry=EmbeddingTelemetry(
            provider=settings.provider,
            model=settings.model,
            call_count=1,
            input_tokens=1,
            duration_seconds=0.0,
            cost_usd=None,
        ),
    )


def test_selected_stages_append_one_canonical_replayable_receipt(
    tmp_path,
    monkeypatch,
):
    calls = []
    observed_prior_evidence = []

    def stage(**arguments):
        calls.append(arguments["stage"])
        observed_prior_evidence.append(dict(arguments["prior_evidence"]))
        return {
            "stage_evidence_id": f"evidence-for-{arguments['stage']}",
        }

    monkeypatch.setattr(smoke_module, "_run_stage", stage)
    selected = ("preflight", "bootstrap-authorities", "github-read")

    first = run_production_smoke(
        config_path=_CONFIG_PATH,
        mode="GENERIC",
        state_root=tmp_path,
        stages=selected,
    )
    replayed = run_production_smoke(
        config_path=_CONFIG_PATH,
        mode="GENERIC",
        state_root=tmp_path,
        stages=selected,
    )

    assert calls == list(selected)
    assert observed_prior_evidence == [
        {},
        {"preflight": {"stage_evidence_id": "evidence-for-preflight"}},
        {
            "preflight": {"stage_evidence_id": "evidence-for-preflight"},
            "bootstrap-authorities": {
                "stage_evidence_id": "evidence-for-bootstrap-authorities"
            },
        },
    ]
    assert replayed == first
    assert tuple(item["stage"] for item in first["stage_receipts"]) == selected
    receipt_path = (
        tmp_path
        / ".kapso/cross_run/production_validation"
        / "production-smoke-receipt.json"
    )
    assert parse_json_bytes(receipt_path.read_bytes()) == first
    assert not (receipt_path.parent / ".production-smoke-receipt.next").exists()


def test_driver_rejects_out_of_order_stage_selection(tmp_path):
    with pytest.raises(ProductionSmokeError, match="out of order"):
        run_production_smoke(
            config_path=_CONFIG_PATH,
            mode="GENERIC",
            state_root=tmp_path,
            stages=("github-read", "preflight"),
        )


def test_driver_fails_loud_on_corrupt_durable_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(
        smoke_module,
        "_run_stage",
        lambda **_arguments: {"passed": True},
    )
    run_production_smoke(
        config_path=_CONFIG_PATH,
        mode="GENERIC",
        state_root=tmp_path,
        stages=("preflight",),
    )
    receipt_path = Path(tmp_path) / (
        ".kapso/cross_run/production_validation/production-smoke-receipt.json"
    )
    receipt_path.write_bytes(b"{not-json}\n")

    with pytest.raises(ValueError):
        run_production_smoke(
            config_path=_CONFIG_PATH,
            mode="GENERIC",
            state_root=tmp_path,
            stages=("preflight",),
        )


def test_synthetic_capture_is_replayable_and_importable(tmp_path):
    settings = load_effective_config(_CONFIG_PATH, "GENERIC").cross_run
    fixture, _fixture_digest = smoke_module._load_fixture(settings)
    scope_contract = smoke_module.ExpertScopeContract.from_dict(
        fixture["scope_contract"]
    )
    capture = smoke_module._synthetic_capture(
        settings,
        fixture,
        scope_contract,
        _EXPERT_RELEASE_ID,
        _TASK_ADAPTER_MANIFEST_ID,
        _TASK_ADAPTER_VERIFICATION_RECEIPT_ID,
    )
    projection = capture.projection
    replayed = RunBundleProjector(settings.capture.score_comparison_tolerance).project(
        capture.stored_bundle
    )
    store = RunBundleStore.initialize(
        tmp_path / settings.capture.state_path,
        settings.capture,
        settings.sanitation,
    )

    assert replayed == projection
    assert store.import_exact(capture.stored_bundle) == capture.stored_bundle
    assert store.require_exact(projection.source_bundle.bundle_id) == (
        capture.stored_bundle
    )
    assert projection.sanitation_report.status == "admitted"
    assert projection.source_bundle.scope_id == "ml_ai"
    assert projection.source_bundle.expert_base_release_id == _EXPERT_RELEASE_ID
    assert (
        projection.source_bundle.artifact_environment.task_adapter_manifest_id
        == _TASK_ADAPTER_MANIFEST_ID
    )
    assert (
        projection.source_bundle.artifact_environment.task_adapter_verification_receipt_id
        == _TASK_ADAPTER_VERIFICATION_RECEIPT_ID
    )
    assert len(projection.episodes) == 1
    assert projection.episodes[0].source_bundle_id == projection.source_bundle.bundle_id
    assert projection.episodes[0].attempts[0].score_of_record_fingerprint_id
    assert projection.episodes[0].attempts[0].intervention_ref is not None
    assert projection.episodes[0].attempts[0].technical_difficulties == (
        "The reusable semantic-parity boundary lacks a common preflight "
        "diagnostic for representation mismatches.",
    )
    assert len(projection.prior_ideas) == 1
    assert projection.catalog_facts[-1] == projection.projection_manifest


@pytest.mark.parametrize("container_type", (list, tuple))
def test_synthetic_projection_accepts_live_and_reloaded_adapter_evidence(
    container_type,
):
    adapter = {
        "task_adapter_id": "posttrain",
        "task_adapter_manifest_id": _TASK_ADAPTER_MANIFEST_ID,
        "verification_receipt_id": _TASK_ADAPTER_VERIFICATION_RECEIPT_ID,
    }

    assert smoke_module._production_task_adapter_pin(
        {
            "task-adapter-bootstrap": {
                "adapters": container_type((adapter,)),
            }
        },
        task_adapter_id="posttrain",
    ) == (_TASK_ADAPTER_MANIFEST_ID, _TASK_ADAPTER_VERIFICATION_RECEIPT_ID)


def test_production_ideation_output_must_cite_the_retrieved_record():
    expected = "transfer-episode:sha256:" + "1" * 64
    output = (
        '{"idea":"change one variable","mechanism":"preserve causality",'
        f'"prior_record_id":"{expected}"}}'
    )

    assert (
        smoke_module._validate_production_ideation_output(
            output,
            expected,
        )["prior_record_id"]
        == expected
    )
    with pytest.raises(ProductionSmokeError, match="selected prior record"):
        smoke_module._validate_production_ideation_output(
            output,
            "transfer-episode:sha256:" + "2" * 64,
        )


def test_expert_bootstrap_exposes_every_scope_task_binding():
    settings = load_effective_config(_CONFIG_PATH, "GENERIC").cross_run
    fixture, _fixture_digest = smoke_module._load_fixture(settings)
    scope_contract = smoke_module.ExpertScopeContract.from_dict(
        fixture["scope_contract"]
    )
    bindings = smoke_module._scope_task_bindings(scope_contract)

    assert tuple(
        (binding.task_family_id, binding.task_adapter_id) for binding in bindings
    ) == (
        ("language_model_post_training", "posttrain"),
        ("relational_tabular_prediction", "relbench"),
    )


def test_task_adapter_bootstrap_precedes_expert_proposal():
    stages = smoke_module.production_smoke_stage_names()

    assert stages.index("task-adapter-bootstrap") < stages.index("expert-proposal")
    assert stages.index("expert-proposal") < stages.index(
        "expert-validation-enrollment"
    )
    assert stages.index("expert-bootstrap-publication") < stages.index(
        "knowledge-publication"
    )
    assert stages.index("coding-agent-ideation") < stages.index(
        "expert-successor-proposal"
    )


def test_production_stage_tail_covers_the_complete_release_lifecycle():
    assert smoke_module.production_smoke_stage_names()[-11:] == (
        "expert-bootstrap-publication",
        "knowledge-publication",
        "coding-agent-ideation",
        "expert-successor-proposal",
        "expert-successor-validation",
        "expert-successor-publication",
        "successor-launch",
        "concurrent-publication",
        "clean-machine-launch",
        "live-restart",
        "revocation",
    )


def test_expert_validation_stage_fails_before_unsigned_evaluator_work(
    tmp_path,
    monkeypatch,
):
    settings = load_effective_config(_CONFIG_PATH, "GENERIC").cross_run
    settings = replace(
        settings,
        expert=replace(
            settings.expert,
            validation=replace(
                settings.expert.validation,
                evaluator_trust_roots=(),
            ),
        ),
    )
    candidate_id = content_id("expert-candidate", {"candidate": "unsigned"})
    snapshot = SimpleNamespace(
        state=SimpleNamespace(
            promotion_state=SimpleNamespace(value="validating"),
            next_stage=smoke_module.ExpertValidationStage.CONTRACT_SCHEMA,
        ),
        transition=SimpleNamespace(transition_id="transition-id"),
    )
    store = SimpleNamespace(snapshot=lambda observed: snapshot)
    monkeypatch.setattr(smoke_module, "_github_services", lambda *_arguments: object())
    monkeypatch.setattr(
        smoke_module,
        "_expert_validation_services",
        lambda *_arguments: SimpleNamespace(validation_store=store),
    )

    def reject_unsigned_result(**_arguments):
        raise AssertionError("unsigned evaluator result reached validation")

    monkeypatch.setattr(
        smoke_module,
        "validate_expert_cross_run",
        reject_unsigned_result,
    )

    with pytest.raises(
        ProductionSmokeError,
        match="externally signed evaluator result.*missing_trust_roots",
    ):
        smoke_module._expert_validation_smoke(
            _CONFIG_PATH,
            "GENERIC",
            settings,
            tmp_path,
            {"expert-validation-enrollment": {"candidate_id": candidate_id}},
            evidence_stage="expert-validation-enrollment",
        )


def test_expert_publication_stage_threads_the_approved_candidate(
    tmp_path,
    monkeypatch,
):
    candidate_id = content_id("expert-candidate", {"candidate": "approved"})
    observed = {}

    def publish(**arguments):
        observed.update(arguments)
        request = parse_json_bytes(arguments["request_path"].read_bytes())
        assert request == {
            "candidate_id": candidate_id,
            "committed_at": "2026-07-20T00:00:00Z",
        }
        return {
            "candidate_id": candidate_id,
            "release_id": "expert-base-release:sha256:" + "1" * 64,
            "activation_receipt_id": "activation-id",
            "publication_id": "publication-id",
            "commit_sha": "1" * 40,
            "release_tag": "expert/E000001",
            "asset_digests": {"expert.tar": "sha256:" + "2" * 64},
            "replayed": False,
        }

    monkeypatch.setattr(smoke_module, "publish_expert_cross_run", publish)
    result = smoke_module._expert_publication_smoke(
        _CONFIG_PATH,
        "GENERIC",
        tmp_path,
        {"committed_at": "2026-07-20T00:00:00Z"},
        {"expert-bootstrap-validation": {"candidate_id": candidate_id}},
        validation_stage="expert-bootstrap-validation",
        proposal_stage="expert-proposal",
    )

    assert observed["state_root"] == tmp_path
    assert result["candidate_id"] == candidate_id
    assert result["publication_skipped"] is False


def test_successor_stage_builds_an_observed_episode_trigger_before_proposal(
    tmp_path,
    monkeypatch,
):
    settings = load_effective_config(_CONFIG_PATH, "GENERIC").cross_run
    trigger_policy = trigger_settings()
    scope_contract, _context, episode, *_rest = source_fixture()
    package, _index, _generation = snapshot_and_index((episode,))
    source_packet = trigger_packet(
        settings=trigger_policy,
        episodes=(episode,),
    )
    release_payload = source_packet.source_base_release.to_dict()
    del release_payload["release_id"]
    release_payload["candidate_tree_hash"] = source_packet.source_base_tree_hash
    release = ExpertBaseReleaseManifest.mint(**release_payload)
    original_receipt = source_packet.source_base_tree_receipt
    extraction = original_receipt.source_extraction_receipt
    source_receipt = ExpertSourceBaseTreeReceipt.mint(
        release_id=release.release_id,
        cache_verification_receipt=replace(
            original_receipt.cache_verification_receipt,
            artifact_id=release.release_id,
        ),
        source_extraction_receipt=SourceArchiveExtractionReceipt.mint(
            artifact_id=release.release_id,
            source_archive_ref=extraction.source_archive_ref,
            source_archive_digest=extraction.source_archive_digest,
            source_tree_hash=extraction.source_tree_hash,
            source_tree_files=extraction.source_tree_files,
            extractor_version=extraction.extractor_version,
        ),
        source_base_tree_hash=original_receipt.source_base_tree_hash,
        repository_map_id=original_receipt.repository_map_id,
        module_contract_ids=original_receipt.module_contract_ids,
        materializer_version=original_receipt.materializer_version,
    )
    module = source_packet.source_base_module_contracts[0]
    configuration_fingerprint = tree_or_blob_digest(
        canonical_json_bytes(trigger_policy.to_dict())
    )
    description = "Add one reusable preflight diagnostic at the capability boundary."
    observation_payload = {
        "affected_capability_ids": [module.module_id],
        "affected_paths": [module.entrypoint_refs[0]],
        "configuration_fingerprint": configuration_fingerprint,
        "description": description,
        "difficulty_evidence_signatures": {},
        "difficulty_signature": None,
        "exact_evidence_ids": [episode.episode_id],
        "independent_lineage_ids": [],
        "inspection_policy_version": trigger_policy.inspection_policy_version,
        "kind": ExpertTriggerObservationKind.MECHANICALLY_GENERAL_FIX.value,
        "occurrence_count": 1,
        "source_base_tree_hash": source_packet.source_base_tree_hash,
        "task_context_binding_ids": [
            episode.task_context_binding.task_context_binding_id
        ],
    }
    inspection_output = canonical_json_bytes(observation_payload).decode("utf-8")
    observation = ExpertTriggerObservation.mint(
        kind=ExpertTriggerObservationKind.MECHANICALLY_GENERAL_FIX,
        source_base_tree_hash=source_packet.source_base_tree_hash,
        inspection_policy_version=trigger_policy.inspection_policy_version,
        configuration_fingerprint=configuration_fingerprint,
        inspection_operation=inspection_operation(
            trigger_policy,
            inspection_output,
        ),
        inspection_final_output=inspection_output,
        difficulty_signature=None,
        difficulty_evidence_signatures={},
        description=description,
        affected_capability_ids=(module.module_id,),
        affected_paths=(module.entrypoint_refs[0],),
        exact_evidence_ids=(episode.episode_id,),
        independent_lineage_ids=(),
        task_context_binding_ids=(
            episode.task_context_binding.task_context_binding_id,
        ),
        occurrence_count=1,
    )
    base = SimpleNamespace(
        release_manifest=release,
        source_base_tree_receipt=source_receipt,
        repository_map=source_packet.source_base_repository_map,
        module_contracts=source_packet.source_base_module_contracts,
        scope_contract=source_packet.source_base_scope_contract,
    )
    base_provider = SimpleNamespace(
        resolve_current=lambda _scope: SimpleNamespace(closure=base)
    )
    github = SimpleNamespace(
        resolver=SimpleNamespace(resolve_current=lambda *_arguments: object()),
        materializer=SimpleNamespace(
            materialize=lambda _resolved: SimpleNamespace(content=tmp_path)
        ),
    )
    monkeypatch.setattr(smoke_module, "_github_services", lambda *_arguments: github)
    monkeypatch.setattr(
        smoke_module,
        "KnowledgeSnapshotPackage",
        SimpleNamespace(open=lambda _content: package),
    )
    monkeypatch.setattr(
        smoke_module,
        "GitHubExpertCompositionBaseProvider",
        lambda *_arguments: base_provider,
    )
    monkeypatch.setattr(
        smoke_module,
        "_inspect_expert_successor_trigger",
        lambda **_arguments: observation,
    )
    observed = {}

    def propose(**arguments):
        request = parse_json_bytes(arguments["request_path"].read_bytes())
        packet = ExpertTriggerEvidencePacket.from_dict(request["evidence_packet"])
        decision = ExpertTriggerEvaluator(settings.expert.triggers).evaluate(packet)
        observed["packet"] = packet
        observed["decision"] = decision
        assert decision.candidate_required is True
        assert decision.reason_code == "mechanically_general_fix"
        return {
            "candidate_id": content_id("expert-candidate", {"stage": "successor"}),
            "candidate_tree_hash": "sha256:" + "1" * 64,
            "change_kind": "capability",
            "proposal_operation_id": "proposal-operation-id",
            "source_base_release_id": release.release_id,
            "trigger_decision_id": decision.trigger_decision_id,
        }

    monkeypatch.setattr(smoke_module, "propose_expert_cross_run", propose)
    result = smoke_module._expert_successor_proposal_smoke(
        _CONFIG_PATH,
        "GENERIC",
        settings,
        tmp_path,
        scope_contract,
        {
            "expert-bootstrap-publication": {
                "release_id": release.release_id,
            }
        },
    )

    assert observed["packet"].episodes == (episode,)
    assert observed["packet"].trigger_observations == (observation,)
    assert result["source_episode_id"] == episode.episode_id
    assert result["trigger_observation_id"] == observation.observation_id


class _ConcurrentBranchClient:
    def __init__(self):
        self.default_head = "a" * 40
        self.branch_head = None
        self.barrier = Barrier(2)
        self.lock = Lock()
        self.updates = []

    def read_ref_commit(self, _repository, qualified_ref, *, allow_missing):
        if qualified_ref == "refs/heads/main":
            return self.default_head
        if self.branch_head is None and not allow_missing:
            raise AssertionError("required branch is absent")
        return self.branch_head

    def create_ref_if_absent(self, _repository, _qualified_ref, commit_sha):
        self.branch_head = commit_sha
        return {"object": {"sha": commit_sha}}

    def api_json(self, method, endpoint, body=None):
        if method == "GET":
            return {"tree": {"sha": "d" * 40}}
        name = body["message"].rsplit(" ", 1)[-1]
        return {"sha": ("b" if name == "alpha" else "c") * 40}

    def update_ref_compare_and_swap(
        self,
        _repository,
        _repository_node_id,
        _branch,
        expected_sha,
        commit_sha,
    ):
        self.barrier.wait()
        with self.lock:
            self.updates.append((expected_sha, commit_sha))
            if self.branch_head != expected_sha:
                raise GitHubCompareAndSwapError("stale parent")
            self.branch_head = commit_sha
            return {"object": {"sha": commit_sha}}


def test_concurrency_smoke_produces_one_winner_and_one_typed_conflict():
    client = _ConcurrentBranchClient()

    result = smoke_module._race_github_publication_branch(
        client,
        repository="Leeroo-AI/kapso-knowledge",
        repository_node_id="repository-node",
        default_branch="main",
        smoke_branch="kapso-production-smoke/concurrent-knowledge",
    )

    assert result["parent_commit_sha"] == "a" * 40
    assert result["winner_commit_sha"] in {"b" * 40, "c" * 40}
    assert result["observed_commit_sha"] == result["winner_commit_sha"]
    assert result["loser_error_type"] == "GitHubCompareAndSwapError"
    assert len(client.updates) == 2
    assert {update[0] for update in client.updates} == {"a" * 40}


def test_live_restart_recomposes_services_and_preserves_pins(
    tmp_path,
    monkeypatch,
):
    settings = load_effective_config(_CONFIG_PATH, "GENERIC").cross_run
    fixture, _fixture_digest = smoke_module._load_fixture(settings)
    scope_contract = smoke_module.ExpertScopeContract.from_dict(
        fixture["scope_contract"]
    )
    closed = []

    class FakePreparedRunHandoff:
        def __init__(self, checkpoint_id):
            self.identity = SimpleNamespace(
                run_id="production-run",
                expert_release_id=_EXPERT_RELEASE_ID,
                knowledge_snapshot_id="knowledge-snapshot:sha256:" + "4" * 64,
            )
            self.frontier = SimpleNamespace(
                checkpoint=SimpleNamespace(run_checkpoint_id=checkpoint_id)
            )
            self.resumed = True

        def close(self):
            closed.append(self.frontier.checkpoint.run_checkpoint_id)

    restart_count = {"value": 0}

    def resume(**_arguments):
        restart_count["value"] += 1
        return FakePreparedRunHandoff(
            "run-checkpoint:sha256:" + str(restart_count["value"]) * 64
        )

    monkeypatch.setattr(
        smoke_module,
        "build_launch_starting_artifact_provider",
        lambda **_arguments: object(),
    )
    monkeypatch.setattr(
        smoke_module,
        "production_experiment_embedding_space",
        lambda _settings: object(),
    )
    monkeypatch.setattr(
        smoke_module,
        "build_production_launch_services",
        lambda **_arguments: SimpleNamespace(coordinator=object()),
    )
    monkeypatch.setattr(smoke_module, "prepare_resumed_run_handoff", resume)
    monkeypatch.setattr(
        smoke_module,
        "_docker_authority_smoke",
        lambda *_arguments: {
            "mutation_lock_device": 1,
            "mutation_lock_inode": 2,
        },
    )
    monkeypatch.setattr(smoke_module, "PreparedRunHandoff", FakePreparedRunHandoff)

    result = smoke_module._live_restart_smoke(
        settings,
        tmp_path,
        scope_contract,
        {
            "clean-machine-launch": {
                "run_id": "production-run",
                "expert_release_id": _EXPERT_RELEASE_ID,
                "knowledge_snapshot_id": ("knowledge-snapshot:sha256:" + "4" * 64),
            }
        },
    )

    assert len(result["service_graph_restarts"]) == 2
    assert result["external_host_restart_performed"] is False
    assert len(closed) == 2


def test_revocation_fences_fresh_launch_and_persisted_resume(
    tmp_path,
    monkeypatch,
):
    settings = load_effective_config(_CONFIG_PATH, "GENERIC").cross_run
    fixture, _fixture_digest = smoke_module._load_fixture(settings)
    scope_contract = smoke_module.ExpertScopeContract.from_dict(
        fixture["scope_contract"]
    )
    checkpoint = SimpleNamespace(
        run_checkpoint_id="run-checkpoint:sha256:" + "5" * 64,
        safety_state=SimpleNamespace(
            security_observation=SimpleNamespace(
                matched_revocations=(
                    SimpleNamespace(
                        revocation_id="security-denylist-revocation:sha256:" + "6" * 64
                    ),
                )
            )
        ),
    )

    class FakeBlockedRunResume:
        def __init__(self):
            self.checkpoint = checkpoint

    def reject_fresh(*_arguments, **_keyword_arguments):
        raise smoke_module.LaunchResolutionError(
            "security denylist rejects the selected launch dependency closure"
        )

    monkeypatch.setattr(smoke_module, "_successor_launch_smoke", reject_fresh)
    monkeypatch.setattr(
        smoke_module,
        "build_launch_starting_artifact_provider",
        lambda **_arguments: object(),
    )
    monkeypatch.setattr(
        smoke_module,
        "production_experiment_embedding_space",
        lambda _settings: object(),
    )
    monkeypatch.setattr(
        smoke_module,
        "build_production_launch_services",
        lambda **_arguments: SimpleNamespace(coordinator=object()),
    )
    monkeypatch.setattr(
        smoke_module,
        "prepare_resumed_run_handoff",
        lambda **_arguments: FakeBlockedRunResume(),
    )
    monkeypatch.setattr(smoke_module, "BlockedRunResume", FakeBlockedRunResume)

    result = smoke_module._verify_revocation_fences(
        config_path=_CONFIG_PATH,
        mode="GENERIC",
        settings=settings,
        smoke_root=tmp_path,
        scope_contract=scope_contract,
        prior_evidence={},
    )

    assert result["fresh_launch_error_type"] == "LaunchResolutionError"
    assert result["fresh_launch_workspace_created"] is False
    assert result["blocked_checkpoint_id"] == checkpoint.run_checkpoint_id
    assert result["persisted_checkpoint_id"] == checkpoint.run_checkpoint_id


def test_trigger_inspection_schema_types_every_fixed_constant():
    fixed = {
        "affected_capability_ids": ["capability"],
        "difficulty_evidence_signatures": {},
        "difficulty_signature": None,
        "occurrence_count": 1,
        "source_base_tree_hash": "sha256:" + "1" * 64,
    }

    schema = smoke_module._expert_trigger_inspection_response_schema(fixed)

    assert schema["properties"] == {
        "affected_capability_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "difficulty_evidence_signatures": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "difficulty_signature": {"type": "null"},
        "occurrence_count": {"type": "integer"},
        "source_base_tree_hash": {"type": "string"},
        "description": {"type": "string", "minLength": 1},
    }


def test_successor_launch_threads_exact_release_snapshot_and_typed_context(
    tmp_path,
    monkeypatch,
):
    settings = load_effective_config(_CONFIG_PATH, "GENERIC").cross_run
    fixture, _fixture_digest = smoke_module._load_fixture(settings)
    scope_contract = smoke_module.ExpertScopeContract.from_dict(
        fixture["scope_contract"]
    )
    context = smoke_module._synthetic_projection(
        settings,
        fixture,
        scope_contract,
        _EXPERT_RELEASE_ID,
        _TASK_ADAPTER_MANIFEST_ID,
        _TASK_ADAPTER_VERIFICATION_RECEIPT_ID,
    ).source_bundle.task_context_binding
    adapter = SimpleNamespace(
        manifest=SimpleNamespace(
            release_matrix_cases=(SimpleNamespace(task_context_binding=context),),
            runtime=SimpleNamespace(to_dict=lambda: {"runtime": "exact"}),
        )
    )
    store = SimpleNamespace(resolve_active=lambda **_arguments: adapter)
    monkeypatch.setattr(smoke_module, "_github_services", lambda *_arguments: object())
    monkeypatch.setattr(
        smoke_module,
        "_expert_validation_services",
        lambda *_arguments: SimpleNamespace(task_adapter_store=store),
    )
    release_id = "expert-base-release:sha256:" + "1" * 64
    snapshot_id = "knowledge-snapshot:sha256:" + "2" * 64

    def resolve(**arguments):
        request = parse_json_bytes(arguments["request_path"].read_bytes())
        parsed_context = smoke_module.LaunchTaskContextRequest.from_dict(
            request["task_context_request"]
        )
        bound_context = parsed_context.bind(
            binding=smoke_module._scope_task_bindings(scope_contract)[0],
            scope_contract=scope_contract,
        )
        assert bound_context.dependency_runtime_fingerprint == tree_or_blob_digest(
            canonical_json_bytes({"runtime": "exact"})
        )
        assert bound_context.transfer_dimensions == context.transfer_dimensions
        assert request["dependency_runtime_contract"] == {"runtime": "exact"}
        return {
            "run_id": "run-id",
            "campaign_id": "campaign-id",
            "launch_manifest_id": "launch-id",
            "bootstrap_pin_id": "pin-id",
            "expert_release_id": release_id,
            "knowledge_snapshot_id": snapshot_id,
            "task_adapter_manifest_id": "adapter-id",
            "workspace_baseline_commit_sha": "3" * 40,
        }

    monkeypatch.setattr(smoke_module, "resolve_launch_cross_run", resolve)
    result = smoke_module._successor_launch_smoke(
        _CONFIG_PATH,
        "GENERIC",
        settings,
        tmp_path,
        scope_contract,
        {
            "expert-successor-publication": {"release_id": release_id},
            "knowledge-publication": {"snapshot_id": snapshot_id},
        },
        clean_machine=False,
    )

    assert result["expert_release_id"] == release_id
    assert result["knowledge_snapshot_id"] == snapshot_id
    assert result["clean_machine"] is False


def test_expert_validation_enrollment_uses_exact_proposal_candidate(
    tmp_path,
    monkeypatch,
):
    smoke_root = tmp_path / "smoke"
    smoke_root.mkdir()
    candidate_id = content_id("expert-candidate", {"candidate": "production"})
    observed = {}

    def validate(**arguments):
        observed.update(arguments)
        request = parse_json_bytes(arguments["request_path"].read_bytes())
        assert request == {
            "candidate_id": candidate_id,
            "evaluator_result": None,
            "expected_transition_id": None,
        }
        return {
            "operation": "validate-expert",
            "candidate_id": candidate_id,
            "validation_attempt_id": "attempt-id",
            "transition_id": "transition-id",
            "validation_state_id": "state-id",
            "next_stage": "contract_schema",
        }

    monkeypatch.setattr(smoke_module, "validate_expert_cross_run", validate)
    result = smoke_module._expert_validation_enrollment_smoke(
        _CONFIG_PATH,
        "GENERIC",
        smoke_root,
        {
            "expert-proposal": {
                "candidate_id": candidate_id,
                "proposal_skipped": False,
            }
        },
    )

    assert observed["config_path"] == _CONFIG_PATH
    assert observed["mode"] == "GENERIC"
    assert observed["state_root"] == smoke_root
    assert result == {
        "candidate_id": candidate_id,
        "validation_attempt_id": "attempt-id",
        "transition_id": "transition-id",
        "validation_state_id": "state-id",
        "next_stage": "contract_schema",
        "validation_skipped": False,
    }


def test_expert_validation_enrollment_skips_authenticated_current_release(tmp_path):
    release_id = content_id("expert-base-release", {"release": "current"})

    assert smoke_module._expert_validation_enrollment_smoke(
        _CONFIG_PATH,
        "GENERIC",
        tmp_path,
        {
            "expert-proposal": {
                "existing_release_id": release_id,
                "proposal_skipped": True,
            }
        },
    ) == {
        "existing_release_id": release_id,
        "validation_skipped": True,
    }


def test_expert_validation_enrollment_requires_proposal_evidence(tmp_path):
    with pytest.raises(ProductionSmokeError, match="proposal evidence"):
        smoke_module._expert_validation_enrollment_smoke(
            _CONFIG_PATH,
            "GENERIC",
            tmp_path,
            {},
        )


def test_preflight_evaluator_summary_exposes_public_roots_without_private_keys():
    settings = load_effective_config(_CONFIG_PATH, "GENERIC").cross_run

    authority = smoke_module._expert_evaluator_authority(settings)

    assert authority["configured"] is True
    assert authority["missing_issuer_ids"] == ()
    assert (
        authority["issuer_trust_roots"]["expert_contract_evaluator"]
        == "kapso_github_evaluator_ed25519_v1"
    )
    assert "expert_source_replay_evaluator" not in authority["issuer_trust_roots"]
    assert "expert_release_matrix_evaluator" not in authority["issuer_trust_roots"]
    assert set(authority) == {
        "configured",
        "issuer_trust_roots",
        "missing_issuer_ids",
        "sealed_canary_trust_root",
    }


@pytest.mark.parametrize(
    ("second_vector_prefix", "passes"),
    (
        ((1.0, 0.001), True),
        ((0.0, 1.0), False),
    ),
)
def test_embedding_smoke_bounds_provider_drift_by_cosine_distance(
    monkeypatch,
    second_vector_prefix,
    passes,
):
    settings = load_effective_config(_CONFIG_PATH, "GENERIC").cross_run
    fixture, _fixture_digest = smoke_module._load_fixture(settings)
    texts = tuple(fixture["embedding_inputs"])
    dimensions = settings.knowledge.embeddings.dimensions
    first_vector = (1.0, 0.0, *(0.0 for _ in range(dimensions - 2)))
    second_vector = (
        *second_vector_prefix,
        *(0.0 for _ in range(dimensions - 2)),
    )
    batches = [
        _embedding_batch(
            settings.knowledge.embeddings,
            texts,
            (first_vector, first_vector),
        ),
        _embedding_batch(
            settings.knowledge.embeddings,
            texts,
            (second_vector, second_vector),
        ),
    ]
    provider = SimpleNamespace(embed=lambda _texts: batches.pop(0))
    monkeypatch.setattr(
        smoke_module,
        "OpenAIEmbeddingProvider",
        lambda _settings: provider,
    )

    if passes:
        evidence = smoke_module._embedding_smoke(settings, fixture)
        assert evidence["maximum_cosine_distance"] <= (
            evidence["cosine_distance_tolerance"]
        )
        assert evidence["first_vector_digest"] != evidence["second_vector_digest"]
    else:
        with pytest.raises(ProductionSmokeError, match="cosine-distance tolerance"):
            smoke_module._embedding_smoke(settings, fixture)


@pytest.mark.parametrize(
    ("configuration_changed", "adapter_changed"),
    ((True, False), (False, True)),
)
def test_clean_root_imports_current_snapshot_and_mints_one_direct_successor(
    tmp_path,
    configuration_changed,
    adapter_changed,
):
    settings = load_effective_config(_CONFIG_PATH, "GENERIC").cross_run
    old_settings = (
        replace(
            settings,
            expert=replace(
                settings.expert,
                architect_id="production_transport_architect_old",
            ),
        )
        if configuration_changed
        else settings
    )
    old_adapter_manifest_id = (
        "task-adapter-manifest:sha256:" + "4" * 64
        if adapter_changed
        else _TASK_ADAPTER_MANIFEST_ID
    )
    fixture, _fixture_digest = smoke_module._load_fixture(settings)
    scope_contract = smoke_module.ExpertScopeContract.from_dict(
        fixture["scope_contract"]
    )
    old_projection = smoke_module._synthetic_projection(
        old_settings,
        fixture,
        scope_contract,
        _EXPERT_RELEASE_ID,
        old_adapter_manifest_id,
        _TASK_ADAPTER_VERIFICATION_RECEIPT_ID,
    )
    old_catalog = CrossRunCatalog(
        tmp_path / "old-catalog",
        scope_contract,
        settings.catalog,
    )
    old_generation = old_catalog.publish_projection(
        old_catalog.store.read_current(),
        old_projection,
    ).generation
    publisher = KnowledgeSnapshotPublisher(
        RecordingPublicationAuthority(),
        settings.github,
        settings.knowledge,
        DeterministicEmbeddingProvider(settings.knowledge.embeddings),
    )
    old_package = publisher.build(
        scope_contract,
        old_generation,
        old_catalog.store.read_object_bytes,
        parent_snapshot_ids=(content_id("knowledge-snapshot", {"transport": "empty"}),),
        sanitation_policy_version=settings.sanitation.policy_version,
        retrieval_policy_version="kapso.retrieval.v1",
        published_at=fixture["committed_at"],
        publisher_attestation={"issuer": "test-publisher"},
    ).package

    successor_capture = smoke_module._synthetic_capture_for_snapshot(
        settings,
        fixture,
        scope_contract,
        old_package,
        _EXPERT_RELEASE_ID,
        _TASK_ADAPTER_MANIFEST_ID,
        _TASK_ADAPTER_VERIFICATION_RECEIPT_ID,
    )
    successor = successor_capture.projection

    if adapter_changed:
        assert successor.source_bundle.capture_generation == 0
        assert successor.source_bundle.supersedes_bundle_id is None
        assert successor.prior_ideas[0].supersedes_projection_id is None
        assert successor.episodes[0].supersedes_projection_id is None
        assert successor.source_bundle.run_id != old_projection.source_bundle.run_id
    else:
        assert successor.source_bundle.capture_generation == 1
        assert successor.source_bundle.supersedes_bundle_id == (
            old_projection.source_bundle.bundle_id
        )
        assert successor.prior_ideas[0].supersedes_projection_id == (
            old_projection.prior_ideas[0].prior_idea_id
        )
        assert successor.episodes[0].supersedes_projection_id == (
            old_projection.episodes[0].episode_id
        )
    new_catalog = CrossRunCatalog(
        tmp_path / "new-catalog",
        scope_contract,
        settings.catalog,
    )
    seeded = smoke_module._seed_catalog_from_snapshot(new_catalog, old_package)
    successor_generation = new_catalog.publish_projection(
        seeded,
        successor,
    ).generation
    successor_package = publisher.build(
        scope_contract,
        successor_generation,
        new_catalog.store.read_object_bytes,
        parent_snapshot_ids=(old_package.manifest.snapshot_id,),
        sanitation_policy_version=settings.sanitation.policy_version,
        retrieval_policy_version="kapso.retrieval.v1",
        published_at=fixture["committed_at"],
        publisher_attestation={"issuer": "test-publisher"},
    ).package

    recovered = smoke_module._synthetic_capture_for_snapshot(
        settings,
        fixture,
        scope_contract,
        successor_package,
        _EXPERT_RELEASE_ID,
        _TASK_ADAPTER_MANIFEST_ID,
        _TASK_ADAPTER_VERIFICATION_RECEIPT_ID,
    )

    assert recovered.projection == successor
    assert recovered.stored_bundle == successor_capture.stored_bundle
