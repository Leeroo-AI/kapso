"""Cross-run ideation keeps foreign knowledge reproducible and advisory."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from kapso.core.embeddings import EmbeddingSettings
from kapso.cross_run.canonical import (
    canonical_json_bytes,
    content_id,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    PriorIdea,
    PriorKnowledgeSnapshot,
    TaskContextBinding,
)
from kapso.cross_run.knowledge.access import PriorKnowledgeAccessMaterialization
from kapso.cross_run.knowledge.retrieval import CrossRunRetriever
from kapso.execution.memories.experiment_memory import ExperimentHistoryStore
from kapso.execution.search_strategies.generic.ideation import (
    IdeaDescriptor,
    IdeaArchive,
    IdeaOutcome,
    IdeationCrossRunIdentity,
    IdeationCrossRunRuntime,
    AnalyzedCandidate,
    CandidateAnalysis,
    CandidateDisposition,
    CandidateDispositionKind,
    EvaluationStatus,
    ImplementationStatus,
    ResurfacedIdea,
    new_identifier,
)
from test_cross_run_retrieval import (
    episode_with_context,
    outcome_episodes,
    relbench_context,
    retrieval_settings,
    snapshot_and_index,
    source_records,
)
from test_ideation_analyzer import analyzer, candidate, context
from test_ideation_engine import (
    CountingParents,
    PacketRunner,
    StoreAheadPacketRunner,
    build_engine,
    run_engine,
)
from test_ideation_evidence import capacity
from test_ideation_domain import (
    OTHER_IDEA_ID,
    coding_agent_call,
    generated_idea,
    planned_batch,
    selection,
)
from test_prior_knowledge_gate import citable_access_materialization


class CountingRetriever(CrossRunRetriever):
    """Real verified retriever with observable query count."""

    def __init__(self, snapshot, search_index, settings):
        super().__init__(snapshot, search_index, settings)
        self.query_count = 0
        self.query_error = None
        self.queries = []

    def retrieve(self, query):
        self.query_count += 1
        self.queries.append(query)
        if self.query_error is not None:
            raise self.query_error
        return super().retrieve(query)


class SemanticCountingRetriever(CountingRetriever):
    def __init__(self, snapshot, search_index, settings, embedding_space_id):
        super().__init__(snapshot, search_index, settings)
        self.embedding_space_id = embedding_space_id

    @property
    def semantic_embedding_space_ids(self):
        return (self.embedding_space_id,)


class ConfiguredQueryEmbedder:
    def __init__(self, model):
        self.settings = EmbeddingSettings(
            enabled=True,
            provider="openai",
            model=model,
            dimensions=2,
            batch_size=1,
            timeout_seconds=5,
            max_retries=0,
            canonicalizer_version="kapso.knowledge_embedding.v1",
        )

    def embed(self, texts):
        raise AssertionError("constructor validation must precede embedding")


class PriorFailureAwareRunner(PacketRunner):
    """Deterministic model boundary whose local proposal responds to a failure."""

    def __init__(self, artifact_dir):
        super().__init__(artifact_dir)

    def run(self, request, response_schema):
        result = super().run(request, response_schema)
        packet = json.loads(request.prompt.split("Mandatory packet:\n\n", 1)[1])
        materialization = packet["foreign_prior_knowledge"]["materialization"]
        negative_ids = ()
        if materialization is not None:
            prior_packet = materialization["prior_knowledge_snapshot"]
            negative_ids = tuple(
                sorted(
                    record_id
                    for record_id, metadata in prior_packet[
                        "selection_metadata"
                    ].items()
                    if metadata["outcome"] == "negative"
                )
            )
        output = json.loads(result.output)
        if request.role == "candidate_0" and negative_ids:
            output["proposal"] = (
                "Replace the failed prior intervention with a guarded local "
                "ablation that isolates its unstable mechanism."
            )
            output["prior_knowledge_refs"] = list(negative_ids)
            output["prior_adaptation_rationale"] = (
                "The prior negative result motivates changing the mechanism and "
                "testing the adapted intervention under current local evidence."
            )
        elif request.role == "candidate_selector" and negative_ids:
            selected = next(
                candidate
                for candidate in packet["local_current_run"]["eligible_candidates"]
                if candidate["prior_knowledge_refs"] == list(negative_ids)
            )
            eligible_ids = tuple(
                candidate["idea_id"]
                for candidate in packet["local_current_run"]["eligible_candidates"]
            )
            output["selected_idea_id"] = selected["idea_id"]
            output["fallback_idea_ids"] = [
                idea_id for idea_id in eligible_ids if idea_id != selected["idea_id"]
            ]
            output["decision_summary"] = (
                "Select the local adaptation that directly tests the prior failure."
            )
            output["prior_knowledge_refs"] = list(negative_ids)
        artifact = Path(result.artifacts[0])
        artifact.write_text(json.dumps(output), encoding="utf-8")
        return replace(result, output=json.dumps(output))


def runtime_fixture():
    scope, task_context, episode, prior_idea, _ = source_records()
    settings = retrieval_settings()
    package, index, _ = snapshot_and_index((episode, prior_idea), settings)
    retriever = CountingRetriever(package, index, settings)
    identity = IdeationCrossRunIdentity(
        launch_manifest_id=content_id("launch-manifest", {"fixture": "ideation"}),
        scope_contract_id=scope.scope_contract_id,
        knowledge_snapshot_id=package.manifest.snapshot_id,
        expert_base_release_id=content_id(
            "expert-base-release",
            {"fixture": "ideation"},
        ),
        embedding_space_id=content_id(
            "embedding-space",
            {"fixture": "ideation"},
        ),
        task_context_binding=task_context,
    )
    return IdeationCrossRunRuntime(identity, retriever, None), retriever, prior_idea


def runtime_with_episode(episode, task_context, scope_id, fixture_name):
    settings = retrieval_settings()
    package, index, _ = snapshot_and_index((episode,), settings)
    identity = IdeationCrossRunIdentity(
        launch_manifest_id=content_id(
            "launch-manifest",
            {"fixture": fixture_name},
        ),
        scope_contract_id=scope_id,
        knowledge_snapshot_id=package.manifest.snapshot_id,
        expert_base_release_id=content_id(
            "expert-base-release",
            {"fixture": fixture_name},
        ),
        embedding_space_id=content_id(
            "embedding-space",
            {"fixture": fixture_name},
        ),
        task_context_binding=task_context,
    )
    return IdeationCrossRunRuntime(
        identity,
        CountingRetriever(package, index, settings),
        None,
    )


def test_prior_failure_changes_selected_local_idea_without_populating_experiment_memory(
    tmp_path,
):
    scope, task_context, episode, _, _ = source_records()
    negative_episode = outcome_episodes(episode)[1]
    incompatible_episode = episode_with_context(
        episode,
        relbench_context(task_context),
        "empty-matched-run",
    )
    cases = (
        (
            "empty",
            runtime_with_episode(
                incompatible_episode,
                task_context,
                scope.scope_contract_id,
                "empty-matched-run",
            ),
        ),
        (
            "negative",
            runtime_with_episode(
                negative_episode,
                task_context,
                scope.scope_contract_id,
                "negative-matched-run",
            ),
        ),
    )
    selected_by_case = {}
    histories = {}
    runners = {}
    for case_name, runtime in cases:
        workspace = tmp_path / case_name
        workspace.mkdir()
        history = ExperimentHistoryStore(
            str(workspace / "experiments.json"),
            objective_direction="maximize",
            require_idea_links=True,
            run_id=f"run_{case_name}",
            campaign_id=f"campaign_{case_name}",
            journal_path=str(workspace / "execution_events.jsonl"),
        )
        assert history.experiments == []
        case_runner = PriorFailureAwareRunner(workspace)
        archive, engine = build_engine(
            workspace,
            case_runner,
            cross_run_runtime=runtime,
        )

        result = run_engine(
            engine,
            CountingParents(workspace),
            experiments=tuple(history.experiments),
        )

        selected_by_case[case_name] = result.selected_idea
        histories[case_name] = history
        runners[case_name] = case_runner
        assert result.selected_idea.idea_id in {
            idea.idea_id for idea in archive.state.ideas
        }

    negative_record_id = negative_episode.episode_id
    assert selected_by_case["empty"].prior_knowledge_refs == ()
    assert selected_by_case["negative"].prior_knowledge_refs == (negative_record_id,)
    assert selected_by_case["empty"].proposal != selected_by_case["negative"].proposal
    assert negative_record_id not in {
        idea.idea_id
        for case_name, _ in cases
        for idea in IdeaArchive(
            tmp_path / case_name / "ideas.json",
            "campaign-alpha",
        ).state.ideas
    }
    assert all(history.experiments == [] for history in histories.values())
    for case_runner in runners.values():
        generator_request = next(
            request for request in case_runner.requests if request.role == "candidate_0"
        )
        generator_packet = json.loads(
            generator_request.prompt.split("Mandatory packet:\n\n", 1)[1]
        )
        assert (
            generator_packet["local_current_run"]["evidence_snapshot"]["experiments"]
            == []
        )


def test_batch_persists_exact_packet_and_resume_never_retrieves_again(tmp_path):
    runtime, retriever, _ = runtime_fixture()
    runner = StoreAheadPacketRunner(tmp_path, fail_once_role="candidate_selector")
    archive, engine = build_engine(
        tmp_path,
        runner,
        cross_run_runtime=runtime,
    )
    parents = CountingParents(tmp_path)

    with pytest.raises(RuntimeError, match="candidate_selector interrupted"):
        run_engine(engine, parents)

    persisted_before_resume = archive.state.batches[0]
    persisted_materialization = persisted_before_resume.prior_knowledge
    assert persisted_before_resume.cross_run_identity == runtime.identity
    assert persisted_materialization is not None
    assert retriever.query_count == 1
    query = retriever.queries[0]
    assert query.problem == "Improve the complete task solution."
    assert query.task_context_binding == runtime.identity.task_context_binding
    assert '"candidate_quota":2' in query.directive

    result = run_engine(
        engine,
        parents,
        resume_batch_id=persisted_before_resume.batch_id,
    )

    persisted_after_resume = archive.state.batches[0]
    assert retriever.query_count == 1
    assert persisted_after_resume.prior_knowledge == persisted_materialization
    assert result.telemetry.prior_retrieval_embedding is None
    assert all(
        request.prior_knowledge == persisted_materialization
        for request in runner.requests
        if request.role.startswith("candidate")
    )


def test_retrieval_failure_creates_no_batch_and_starts_no_agent(tmp_path):
    runtime, retriever, _ = runtime_fixture()
    retriever.query_error = RuntimeError("retrieval unavailable")
    runner = StoreAheadPacketRunner(tmp_path, fail_once_role="unused")
    archive, engine = build_engine(
        tmp_path,
        runner,
        cross_run_runtime=runtime,
    )

    with pytest.raises(RuntimeError, match="retrieval unavailable"):
        run_engine(engine, CountingParents(tmp_path))

    assert retriever.query_count == 1
    assert archive.state.batches == ()
    assert archive.state.ideas == ()
    assert runner.roles == []


def test_foreign_exact_match_is_advisory_and_cannot_become_local_authority(tmp_path):
    runtime, _, prior_idea = runtime_fixture()
    retrieval = runtime.retrieve(
        problem_statement="Improve the complete task solution.",
        evidence_snapshot=context(tmp_path)[1],
        directive=context(tmp_path)[2],
    ).retrieval
    materialization = retrieval.access_materialization
    selected_ids = materialization.prior_knowledge_snapshot.selected_record_ids
    assert prior_idea.prior_idea_id in selected_ids

    archive, snapshot, search_directive = context(tmp_path)
    descriptor = IdeaDescriptor.from_dict(dict(prior_idea.descriptor))
    brief = replace(
        search_directive.operator_briefs[0],
        descriptor_target=descriptor,
    )
    search_directive = replace(search_directive, operator_briefs=(brief,))
    local_candidate = candidate(
        tmp_path,
        snapshot,
        search_directive,
        proposal=prior_idea.proposal,
        descriptor=descriptor,
    )

    analyzed = (
        analyzer()
        .analyze_pool(
            batch_id=local_candidate.origin_batch_id,
            candidates=(local_candidate,),
            archive_state=archive.state,
            evidence_snapshot=snapshot,
            directive=search_directive,
            capacity=capacity(),
            prior_knowledge=materialization,
        )
        .candidates[0]
    )

    match = next(
        item
        for item in analyzed.analysis.prior_knowledge_matches
        if item.record_id == prior_idea.prior_idea_id
    )
    assert match.exact_match
    assert analyzed.analysis.eligible
    assert analyzed.analysis.exact_duplicate_of is None
    assert f"prior_adaptation_missing:{prior_idea.prior_idea_id}" in (
        analyzed.similarity_flags
    )
    assert analyzed.nearest_experiment_node_ids == ()


def test_bridge_persists_an_explicit_empty_packet_for_incompatible_knowledge(
    tmp_path,
):
    scope, task_context, episode, _, _ = source_records()
    incompatible = episode_with_context(
        episode,
        relbench_context(task_context),
        "incompatible-bridge",
    )
    settings = retrieval_settings()
    package, index, _ = snapshot_and_index((incompatible,), settings)
    retriever = CountingRetriever(package, index, settings)
    identity = IdeationCrossRunIdentity(
        launch_manifest_id=content_id(
            "launch-manifest",
            {"fixture": "empty-bridge"},
        ),
        scope_contract_id=scope.scope_contract_id,
        knowledge_snapshot_id=package.manifest.snapshot_id,
        expert_base_release_id=content_id(
            "expert-base-release",
            {"fixture": "empty-bridge"},
        ),
        embedding_space_id=content_id(
            "embedding-space",
            {"fixture": "empty-bridge"},
        ),
        task_context_binding=task_context,
    )
    runtime = IdeationCrossRunRuntime(identity, retriever, None)
    _, evidence_snapshot, search_directive = context(tmp_path)

    result = runtime.retrieve(
        problem_statement="Improve the complete task solution.",
        evidence_snapshot=evidence_snapshot,
        directive=search_directive,
    )

    packet = result.retrieval.access_materialization.prior_knowledge_snapshot
    assert packet.selected_record_ids == ()
    assert packet.selected_records == ()
    assert packet.proof_reference_ids == ()
    assert packet.prior_knowledge_snapshot_id.startswith(
        "prior-knowledge-snapshot:sha256:"
    )
    assert retriever.query_count == 1


def test_bridge_rejects_launch_and_query_embedder_space_drift():
    scope, task_context, episode, _, _ = source_records()
    settings = retrieval_settings()
    package, index, _ = snapshot_and_index((episode,), settings)
    correct_embedder = ConfiguredQueryEmbedder("semantic-correct")
    other_embedder = ConfiguredQueryEmbedder("semantic-other")
    correct_space = correct_embedder.settings.embedding_space_id.value
    retriever = SemanticCountingRetriever(
        package,
        index,
        settings,
        correct_space,
    )
    base_identity = IdeationCrossRunIdentity(
        launch_manifest_id=content_id(
            "launch-manifest",
            {"fixture": "semantic-pin"},
        ),
        scope_contract_id=scope.scope_contract_id,
        knowledge_snapshot_id=package.manifest.snapshot_id,
        expert_base_release_id=content_id(
            "expert-base-release",
            {"fixture": "semantic-pin"},
        ),
        embedding_space_id=correct_space,
        task_context_binding=task_context,
    )

    with pytest.raises(ValueError, match="query embedder differs"):
        IdeationCrossRunRuntime(base_identity, retriever, other_embedder)
    with pytest.raises(ValueError, match="launch embedding space is absent"):
        IdeationCrossRunRuntime(
            replace(
                base_identity,
                embedding_space_id=other_embedder.settings.embedding_space_id.value,
            ),
            retriever,
            other_embedder,
        )


def test_cross_run_identity_rejects_duplicate_active_exclusions():
    runtime, _, _ = runtime_fixture()

    with pytest.raises(ValueError, match="must not contain duplicates"):
        replace(
            runtime.identity,
            active_exclusions=("unsafe-pattern", "unsafe-pattern"),
        )


def test_unknown_foreign_reference_is_a_local_candidate_hard_failure(tmp_path):
    runtime, _, _ = runtime_fixture()
    archive, snapshot, search_directive = context(tmp_path)
    materialization = runtime.retrieve(
        problem_statement="Improve the complete task solution.",
        evidence_snapshot=snapshot,
        directive=search_directive,
    ).retrieval.access_materialization
    local_candidate = candidate(
        tmp_path,
        snapshot,
        search_directive,
        prior_knowledge_refs=(content_id("prior-idea", {"unknown": True}),),
        prior_adaptation_rationale="Adapt the cited mechanism to this task.",
    )

    analyzed = (
        analyzer()
        .analyze_pool(
            batch_id=local_candidate.origin_batch_id,
            candidates=(local_candidate,),
            archive_state=archive.state,
            evidence_snapshot=snapshot,
            directive=search_directive,
            capacity=capacity(),
            prior_knowledge=materialization,
        )
        .candidates[0]
    )

    assert not analyzed.analysis.eligible
    assert "prior_knowledge_reference_unknown" in analyzed.analysis.hard_failures


def test_deferred_prior_derived_idea_resurfaces_without_rebinding_origin_refs(
    tmp_path,
):
    origin_materialization = citable_access_materialization()
    origin_packet = origin_materialization.prior_knowledge_snapshot
    prior_record_id = origin_packet.selected_record_ids[0]
    task_context = TaskContextBinding.from_dict(
        origin_packet.selected_records[0]["payload"]["task_context_binding"]
    )
    identity = IdeationCrossRunIdentity(
        launch_manifest_id=content_id("launch-manifest", {"fixture": "resurface"}),
        scope_contract_id=task_context.scope_contract_id,
        knowledge_snapshot_id=origin_packet.source_snapshot_id,
        expert_base_release_id=content_id(
            "expert-base-release",
            {"fixture": "resurface"},
        ),
        embedding_space_id=content_id(
            "embedding-space",
            {"fixture": "resurface"},
        ),
        task_context_binding=task_context,
    )
    empty_packet = PriorKnowledgeSnapshot.mint(
        source_snapshot_id=origin_packet.source_snapshot_id,
        query={"problem": "A later query with no eligible foreign records."},
        retrieval_policy_version=origin_packet.retrieval_policy_version,
        task_context_binding_id=origin_packet.task_context_binding_id,
        selected_records=(),
        selected_record_ids=(),
        proof_reference_ids=(),
        selection_metadata={},
        prompt_budget_policy=origin_packet.prompt_budget_policy,
        records_digest=tree_or_blob_digest(canonical_json_bytes(())),
    )
    empty_materialization = PriorKnowledgeAccessMaterialization.mint(
        prior_knowledge_snapshot=empty_packet,
        proof_records=(),
    )
    archive = IdeaArchive(tmp_path / "ideas.json", "campaign-alpha")
    origin_batch_id = new_identifier("batch")
    base_batch = planned_batch()
    origin_directive = replace(
        base_batch.directive,
        operator_briefs=base_batch.directive.operator_briefs * 2,
        candidate_quota=2,
    )
    origin_batch = replace(
        base_batch,
        batch_id=origin_batch_id,
        context_hash="4" * 64,
        directive=origin_directive,
        resolved_parents=base_batch.resolved_parents * 2,
        cross_run_identity=identity,
        prior_knowledge=origin_materialization,
    )
    deferred = replace(
        generated_idea(),
        origin_batch_id=origin_batch_id,
        prior_knowledge_refs=(prior_record_id,),
        prior_adaptation_rationale="Adapt the prior mechanism to local evidence.",
    )
    selected_first = replace(
        generated_idea(OTHER_IDEA_ID),
        origin_batch_id=origin_batch_id,
        proposal="Measure an independent local control first.",
    )
    archive.create_batch(origin_batch, expected_revision=0)
    archive.add_ideas(
        origin_batch_id,
        (deferred, selected_first),
        generation_calls=(coding_agent_call(), coding_agent_call()),
        expected_revision=archive.revision,
    )
    archive.record_analyses(
        origin_batch_id,
        tuple(
            AnalyzedCandidate(
                analysis=CandidateAnalysis(idea_id=idea.idea_id, eligible=True),
                descriptor=idea.descriptor,
                embedding=None,
                nearest_experiment_node_ids=(),
                similarity_flags=(),
            )
            for idea in (deferred, selected_first)
        ),
        expected_revision=archive.revision,
    )
    archive.record_selection(
        origin_batch_id,
        replace(
            selection(),
            selected_idea_id=selected_first.idea_id,
            fallback_idea_ids=(deferred.idea_id,),
            dispositions=(
                CandidateDisposition(
                    deferred.idea_id,
                    CandidateDispositionKind.DEFERRED,
                    "Retain for a changed evidence state.",
                ),
                CandidateDisposition(
                    selected_first.idea_id,
                    CandidateDispositionKind.SELECTED,
                    "Run the local control first.",
                ),
            ),
            prior_knowledge_refs=(),
        ),
        selection_call=coding_agent_call(),
        expected_revision=archive.revision,
    )
    archive.link_experiment(
        selected_first.idea_id,
        0,
        origin_batch_id,
        expected_revision=archive.revision,
    )
    archive.record_outcome(
        selected_first.idea_id,
        IdeaOutcome(
            evaluation_status=EvaluationStatus.VALID,
            implementation_status=ImplementationStatus.COMPLETED,
            normalized_delta=0.0,
            validation_tier="full",
            actual_cost=1.0,
            actual_duration=30.0,
        ),
        expected_revision=archive.revision,
    )

    resurface_batch_id = new_identifier("batch")
    resurface_batch = replace(
        base_batch,
        batch_id=resurface_batch_id,
        iteration_index=1,
        context_hash="5" * 64,
        planning_archive_revision=archive.revision,
        cross_run_identity=identity,
        prior_knowledge=empty_materialization,
    )
    archive.create_batch(resurface_batch, expected_revision=archive.revision)
    archive.add_ideas(
        resurface_batch_id,
        (),
        generation_calls=(),
        resurfaced_ideas=(
            ResurfacedIdea(
                idea_id=deferred.idea_id,
                changed_conditions=("the local control is now measured",),
            ),
        ),
        expected_revision=archive.revision,
    )
    archive.record_analyses(
        resurface_batch_id,
        (
            AnalyzedCandidate(
                analysis=CandidateAnalysis(
                    idea_id=deferred.idea_id,
                    eligible=True,
                ),
                descriptor=deferred.descriptor,
                embedding=None,
                nearest_experiment_node_ids=(),
                similarity_flags=(),
            ),
        ),
        expected_revision=archive.revision,
    )
    archive.record_selection(
        resurface_batch_id,
        replace(
            selection(),
            selected_idea_id=deferred.idea_id,
            dispositions=(
                CandidateDisposition(
                    deferred.idea_id,
                    CandidateDispositionKind.SELECTED,
                    "Changed local evidence now favors the deferred idea.",
                ),
            ),
            prior_knowledge_refs=(),
        ),
        selection_call=coding_agent_call(),
        expected_revision=archive.revision,
    )

    persisted = archive.get_idea(deferred.idea_id)
    assert persisted.prior_knowledge_refs == (prior_record_id,)
    assert persisted.selected_in_batch_id == resurface_batch_id
    assert archive.get_batch(resurface_batch_id).prior_knowledge == (
        empty_materialization
    )
