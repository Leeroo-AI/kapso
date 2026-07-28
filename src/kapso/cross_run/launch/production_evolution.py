"""Public GitHub-backed evolution composed from the verified launch boundary."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

from kapso.core.embedding_contracts import (
    EmbeddingSettings as ProviderEmbeddingSettings,
)
from kapso.core.embedding_provider import OpenAIEmbeddingProvider
from kapso.cross_run.canonical import canonical_json_bytes, tree_or_blob_digest
from kapso.cross_run.capture.bundle import RunBundleStore
from kapso.cross_run.docker.runtime import DockerImageAuthority
from kapso.cross_run.knowledge.index import SnapshotSearchIndex
from kapso.cross_run.knowledge.package import KnowledgeSnapshotPackage
from kapso.cross_run.knowledge.retrieval import (
    CrossRunRetriever,
    PriorKnowledgeQuery,
)
from kapso.cross_run.launch.contracts import LaunchTaskContextRequest
from kapso.cross_run.launch.handoff import (
    PreparedRunHandoff,
    prepare_fresh_run_handoff,
    prepare_run_action_recovery_handoff,
    prepare_resumed_run_handoff,
)
from kapso.cross_run.launch.production import (
    build_production_launch_preparation,
    build_production_launch_services,
    production_experiment_embedding_space,
    ProductionLaunchServices,
    resolve_production_binding,
)
from kapso.cross_run.launch.resume import BlockedRunResume
from kapso.cross_run.launch.resume_contracts import RunReleaseUseMode
from kapso.cross_run.launch.run_action_coding_agent_contracts import (
    CODING_AGENT_REQUEST_PROTOCOL_VERSION,
    CodingAgentRunActionRequest,
)
from kapso.cross_run.launch.run_action_coding_agent_schema import (
    CODING_AGENT_JSON_SCHEMA_DIALECT,
)
from kapso.cross_run.launch.run_action_coding_agent_service import (
    build_production_coding_agent_action,
    ProductionCodingAgentActionResult,
)
from kapso.cross_run.launch.run_action_contracts import (
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_ledger import (
    RunActionLedgerSnapshot,
    RunActionOperationTail,
)
from kapso.cross_run.launch.starting_artifacts import (
    LaunchStartingArtifactSetProvider,
)
from kapso.cross_run.launch.workspace_frontier import (
    inspect_run_workspace_frontier,
)
from kapso.cross_run.settings import CrossRunSettings, EffectiveConfig

_PRIOR_KNOWLEDGE_DIRECTIVE = (
    "Retrieve tested interventions, failures, contradictions, and frontier ideas "
    "that can inform one novel but evidence-grounded experiment for this task."
)
_EVOLUTION_PRINCIPAL_ID = "kapso.evolution_agent"
_EVOLUTION_ROLE = "ideation_implementation"
_EVOLUTION_RESPONSE_SCHEMA = {
    "$schema": CODING_AGENT_JSON_SCHEMA_DIALECT,
    "type": "object",
    "properties": {
        "idea_summary": {"type": "string", "minLength": 1},
        "exploration_rationale": {"type": "string", "minLength": 1},
        "implementation_summary": {"type": "string", "minLength": 1},
        "validation_summary": {"type": "string", "minLength": 1},
        "prior_knowledge_record_ids": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "uncertainties": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
    },
    "required": [
        "idea_summary",
        "exploration_rationale",
        "implementation_summary",
        "validation_summary",
        "prior_knowledge_record_ids",
        "uncertainties",
    ],
    "additionalProperties": False,
}


class ProductionEvolutionError(RuntimeError):
    """The sole public evolution path cannot preserve its pinned authority."""


@dataclass(frozen=True)
class ProductionEvolutionResult:
    """One accepted direct-successor experiment and its public identities."""

    code_path: Path
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.code_path, Path)
            or not self.code_path.is_absolute()
            or not isinstance(self.metadata, Mapping)
        ):
            raise ProductionEvolutionError("production evolution result is invalid")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def execute_production_evolution(
    *,
    effective_config: EffectiveConfig,
    goal: str,
    run_root: Path,
    state_root: Path,
    task_context_request: LaunchTaskContextRequest | None,
    starting_artifact_sources: Mapping[str, tuple[Path, str]],
    dependency_runtime_contract: Mapping[str, Any] | None,
    budget_fidelity_envelope: Mapping[str, Any] | None,
    scope_id: str | None,
    task_family_id: str | None,
    task_adapter_id: str | None,
    requested_coding_agent: str | None,
    objective_direction: str,
    additional_context: str,
    resume: bool,
    empty_scope_bootstrap_authorization_id: str | None = None,
) -> ProductionEvolutionResult:
    """Prepare the launch before any paid call, then retrieve and edit once."""

    if (
        type(effective_config) is not EffectiveConfig
        or not isinstance(goal, str)
        or not goal.strip()
        or not isinstance(run_root, Path)
        or not run_root.is_absolute()
        or not isinstance(state_root, Path)
        or not state_root.is_absolute()
        or not isinstance(starting_artifact_sources, Mapping)
        or not isinstance(additional_context, str)
        or type(resume) is not bool
        or objective_direction not in {"maximize", "minimize"}
    ):
        raise ProductionEvolutionError("production evolution inputs are invalid")
    settings = effective_config.cross_run
    if settings is None:
        raise ProductionEvolutionError(
            "selected configuration has no cross-run settings"
        )
    binding = resolve_production_binding(
        effective_config=effective_config,
        scope_id=scope_id,
        task_family_id=task_family_id,
        task_adapter_id=task_adapter_id,
    )
    image_authority = _configured_image_authority(effective_config)
    if requested_coding_agent not in {None, settings.launch.coding_agent.cli}:
        raise ProductionEvolutionError(
            "requested coding agent differs from launch coding-agent authority"
        )

    if resume:
        if (
            task_context_request is not None
            or starting_artifact_sources
            or dependency_runtime_contract is not None
            or budget_fidelity_envelope is not None
            or empty_scope_bootstrap_authorization_id is not None
        ):
            raise ProductionEvolutionError(
                "resume accepts only the pinned local launch inputs"
            )
        experiment_embedding_space = production_experiment_embedding_space(settings)
        starting_artifacts = LaunchStartingArtifactSetProvider((), settings.launch)
        request = None
    else:
        if (
            type(task_context_request) is not LaunchTaskContextRequest
            or dependency_runtime_contract is None
            or budget_fidelity_envelope is None
        ):
            raise ProductionEvolutionError(
                "fresh evolution requires task context, runtime, and budget authority"
            )
        preparation = build_production_launch_preparation(
            effective_config=effective_config,
            goal=goal,
            additional_context=additional_context,
            task_context_request=task_context_request,
            starting_artifact_sources=starting_artifact_sources,
            dependency_runtime_contract=dependency_runtime_contract,
            budget_fidelity_envelope=budget_fidelity_envelope,
            scope_id=scope_id,
            task_family_id=task_family_id,
            task_adapter_id=task_adapter_id,
            requested_coding_agent=settings.launch.coding_agent.cli,
            empty_scope_bootstrap_authorization_id=(
                empty_scope_bootstrap_authorization_id
            ),
        )
        binding = preparation.binding
        experiment_embedding_space = preparation.experiment_embedding_space
        starting_artifacts = preparation.starting_artifacts
        request = preparation.request

    services = build_production_launch_services(
        settings=settings,
        binding=binding,
        experiment_embedding_space=experiment_embedding_space,
        starting_artifacts=starting_artifacts,
        state_root=state_root,
    )
    recovered_action = (
        _recover_resumed_action(
            services=services,
            settings=settings,
            image_authority=image_authority,
            run_root=run_root,
            state_root=state_root,
            goal=goal,
            additional_context=additional_context,
        )
        if resume
        else None
    )
    with ExitStack() as resources:
        if resume:
            resumed = prepare_resumed_run_handoff(
                coordinator=services.coordinator,
                settings=settings,
                run_root=run_root,
                release_use_mode=RunReleaseUseMode.ONLINE_CURRENT,
            )
            if type(resumed) is BlockedRunResume:
                raise ProductionEvolutionError(
                    "resume is blocked by current security or release-use policy"
                )
            if type(resumed) is not PreparedRunHandoff:
                raise ProductionEvolutionError("resume returned unknown authority")
            handoff = resumed
        else:
            if request is None:
                raise ProductionEvolutionError("fresh launch request is absent")
            handoff = prepare_fresh_run_handoff(
                coordinator=services.coordinator,
                settings=settings,
                security_authority=services.security_authority,
                request=request,
                run_root=run_root,
                objective_direction=objective_direction,
            )
            RunBundleStore.initialize(
                state_root / settings.capture.state_path,
                settings.capture,
                settings.sanitation,
            ).publish_starting_artifacts(
                task_context_binding=(
                    handoff.active_workspace.bootstrap_pin.launch_manifest.task_context_binding
                ),
                launch_artifacts=starting_artifacts.artifacts,
                validation_settings=settings.expert.validation,
            )
        _require_pinned_prompt_inputs(
            handoff.active_workspace.bootstrap_pin.launch_manifest.launch_request,
            goal=goal,
            additional_context=additional_context,
        )
        resources.callback(handoff.close)

        embedding_telemetry = None
        if recovered_action is None:
            prior_knowledge, embedding_telemetry = _retrieve_prior_knowledge(
                handoff=handoff,
                goal=goal,
                settings=settings,
            )
            predecessor_source_tree_digest = _source_tree_digest(handoff, settings)
            prompt = _evolution_prompt(
                goal=goal,
                additional_context=additional_context,
                repository_memory=handoff.repository_memory.payload,
            )
            action = build_production_coding_agent_action(
                handoff=handoff,
                services=services,
                settings=settings,
                image_authority=image_authority,
                agent=settings.launch.coding_agent,
                state_root=state_root,
                principal_id=_EVOLUTION_PRINCIPAL_ID,
                role=_EVOLUTION_ROLE,
                workspace_access=RunFrontierWorkspaceAccess.EDIT_WORKSPACE,
                web_search_enabled=settings.launch.coding_agent_web_search_enabled,
                provider_network_enabled=(
                    settings.launch.coding_agent_provider_network_enabled
                ),
                native_credential_enabled=True,
            )
            resources.callback(action.close)
            operation_id = _operation_id(
                handoff=handoff,
                prompt=prompt,
                predecessor_source_tree_digest=predecessor_source_tree_digest,
            )
            action_request = CodingAgentRunActionRequest(
                protocol_version=CODING_AGENT_REQUEST_PROTOCOL_VERSION,
                interpretation_policy=action.interpretation_policy,
                operation_id=operation_id,
                prompt=prompt,
                response_schema=_EVOLUTION_RESPONSE_SCHEMA,
                prior_knowledge=prior_knowledge,
                edit_predecessor_source_tree_digest=predecessor_source_tree_digest,
            )
            action_result = action.execute(
                frontier=handoff.frontier,
                request=action_request,
            )
        else:
            action_result = recovered_action
            action_request = recovered_action.request
            operation_id = action_request.operation_id
        if action_request.prior_knowledge is None:
            raise ProductionEvolutionError(
                "production evolution action lacks pinned prior knowledge"
            )
        identity = handoff.identity
        packet = action_request.prior_knowledge.prior_knowledge_snapshot
        return ProductionEvolutionResult(
            code_path=handoff.active_workspace.workspace,
            metadata={
                "run_id": identity.run_id,
                "campaign_id": identity.campaign_id,
                "launch_manifest_id": identity.launch_manifest_id,
                "bootstrap_pin_id": identity.bootstrap_pin_id,
                "scope_id": identity.scope_id,
                "scope_contract_id": identity.scope_contract_id,
                "task_family_id": identity.task_family_id,
                "task_adapter_id": identity.task_adapter_id,
                "task_context_binding_id": identity.task_context_binding_id,
                "expert_release_id": identity.expert_release_id,
                "knowledge_snapshot_id": identity.knowledge_snapshot_id,
                "task_adapter_manifest_id": identity.task_adapter_manifest_id,
                "task_adapter_activation_id": identity.task_adapter_activation_id,
                "repository_memory_digest": handoff.repository_memory.digest,
                "operation_id": operation_id,
                "action_request_digest": action_request.request_digest,
                "action_result": dict(action_result.result.structured_output),
                "action_input_tokens": action_result.result.input_tokens,
                "action_output_tokens": action_result.result.output_tokens,
                "action_cost_usd": action_result.result.cost_usd,
                "prior_knowledge_packet_id": packet.prior_knowledge_snapshot_id,
                "prior_knowledge_record_ids": packet.selected_record_ids,
                "prior_knowledge_access_count": len(
                    action_result.result.prior_knowledge_accesses
                ),
                "prior_retrieval_embedding_telemetry": (
                    None
                    if embedding_telemetry is None
                    else embedding_telemetry.to_dict()
                ),
                "resumed": resume,
                "frontier_checkpoint_id": (
                    (
                        handoff.frontier
                        if recovered_action is not None
                        else action_result.frontier
                    ).checkpoint.run_checkpoint_id
                ),
            },
        )


def _recover_resumed_action(
    *,
    services: ProductionLaunchServices,
    settings: CrossRunSettings,
    image_authority: DockerImageAuthority,
    run_root: Path,
    state_root: Path,
    goal: str,
    additional_context: str,
) -> ProductionCodingAgentActionResult | None:
    """Recover the sole durable evolution action before normal run resume."""

    with ExitStack() as resources:
        handoff = prepare_run_action_recovery_handoff(
            coordinator=services.coordinator,
            settings=settings,
            run_root=run_root,
        )
        resources.callback(handoff.close)
        _require_pinned_prompt_inputs(
            handoff.active_workspace.bootstrap_pin.launch_manifest.launch_request,
            goal=goal,
            additional_context=additional_context,
        )
        projected_ledger = handoff.frontier.projection.action_ledger
        live_ledger = handoff.publisher.action_ledger_snapshot()
        if live_ledger == projected_ledger:
            return None
        unprojected_tails = _unprojected_action_tails(
            projected_ledger,
            live_ledger,
        )
        if len(unprojected_tails) != 1:
            raise ProductionEvolutionError(
                "production evolution run contains multiple unprojected actions"
            )
        action = build_production_coding_agent_action(
            handoff=handoff,
            services=services,
            settings=settings,
            image_authority=image_authority,
            agent=settings.launch.coding_agent,
            state_root=state_root,
            principal_id=_EVOLUTION_PRINCIPAL_ID,
            role=_EVOLUTION_ROLE,
            workspace_access=RunFrontierWorkspaceAccess.EDIT_WORKSPACE,
            web_search_enabled=settings.launch.coding_agent_web_search_enabled,
            provider_network_enabled=(
                settings.launch.coding_agent_provider_network_enabled
            ),
            native_credential_enabled=True,
        )
        resources.callback(action.close)
        return action.recover_existing(
            frontier=handoff.frontier,
            operation_id=unprojected_tails[0].operation_id,
        )


def _unprojected_action_tails(
    projected: RunActionLedgerSnapshot,
    live: RunActionLedgerSnapshot,
) -> tuple[RunActionOperationTail, ...]:
    """Return only durable operations not yet consumed by the checkpoint."""

    if (
        type(projected) is not RunActionLedgerSnapshot
        or type(live) is not RunActionLedgerSnapshot
    ):
        raise ProductionEvolutionError(
            "production evolution action ledgers are invalid"
        )
    live.require_predecessor(projected)
    projected_tails = {tail.operation_id: tail for tail in projected.operation_tails}
    return tuple(
        tail
        for tail in live.operation_tails
        if projected_tails.get(tail.operation_id) != tail
    )


def _configured_image_authority(
    effective_config: EffectiveConfig,
) -> DockerImageAuthority:
    settings = effective_config.cross_run
    if settings is None or settings.launch.coding_agent_image is None:
        raise ProductionEvolutionError(
            "runtime config lacks a pinned launch.coding_agent_image authority"
        )
    image = settings.launch.coding_agent_image
    return DockerImageAuthority.mint(
        image_reference=image.image_reference,
        image_config_digest=image.image_config_digest,
        operating_system=image.operating_system,
        architecture=image.architecture,
        architecture_variant=image.architecture_variant,
    )


def _retrieve_prior_knowledge(*, handoff, goal, settings):
    """Open only the pinned package and embed only after prepared handoff."""

    if type(handoff) is not PreparedRunHandoff:
        raise ProductionEvolutionError("prior knowledge requires prepared handoff")
    handoff.active_workspace.require_control_authority()
    manifest = handoff.active_workspace.bootstrap_pin.launch_manifest
    layout = handoff.active_workspace.bootstrap_pin.installation_receipt.layout
    package_root = handoff.active_workspace.run_root / (
        layout.knowledge_snapshot_relative_path
    )
    package = KnowledgeSnapshotPackage.open(package_root)
    if package.manifest != manifest.knowledge_manifest:
        raise ProductionEvolutionError("pinned knowledge package identity changed")
    index_files = {
        path: payload
        for path, payload in package.files.items()
        if PurePosixPath(path).parts[0] == "index"
    }
    if not index_files:
        raise ProductionEvolutionError("pinned knowledge package lacks search index")
    search_index = SnapshotSearchIndex.open(package.prepared, index_files)
    retriever = CrossRunRetriever(package, search_index, settings.knowledge.retrieval)
    query = PriorKnowledgeQuery(
        task_context_binding=manifest.task_context_binding,
        problem=goal,
        current_gaps=(),
        directive=_PRIOR_KNOWLEDGE_DIRECTIVE,
    )
    telemetry = None
    if retriever.semantic_embedding_space_ids:
        configured = settings.knowledge.embeddings
        provider_settings = ProviderEmbeddingSettings(
            enabled=configured.enabled,
            provider=configured.provider,
            model=configured.model,
            dimensions=configured.dimensions,
            batch_size=configured.batch_size,
            timeout_seconds=configured.timeout_seconds,
            max_retries=configured.max_retries,
            canonicalizer_version=configured.canonicalizer_version,
        )
        embedding = OpenAIEmbeddingProvider(provider_settings).embed(
            (query.lexical_text,)
        )
        if len(embedding.records) != 1:
            raise ProductionEvolutionError(
                "prior-knowledge query embedder returned an invalid batch"
            )
        query = PriorKnowledgeQuery(
            task_context_binding=query.task_context_binding,
            problem=query.problem,
            current_gaps=query.current_gaps,
            directive=query.directive,
            query_embedding=embedding.records[0],
        )
        telemetry = embedding.telemetry
    retrieval = retriever.retrieve(query)
    return retrieval.access_materialization, telemetry


def _source_tree_digest(handoff, settings) -> str:
    branch = settings.launch.workspace_git_branch
    expected_commit = handoff.frontier.checkpoint.safety_state.derivative_frontier.evidence.branch_heads.get(
        branch
    )
    if expected_commit is None:
        raise ProductionEvolutionError("run frontier omits its workspace branch")
    with ExitStack() as descriptors:
        workspace_descriptor, _identity = (
            handoff.active_workspace._open_execution_workspace(descriptors)
        )
        frontier = inspect_run_workspace_frontier(
            workspace_descriptor,
            settings=settings.launch,
            expected_commit_sha=expected_commit,
        )
    return frontier.source_tree_digest


def _evolution_prompt(
    *,
    goal: str,
    additional_context: str,
    repository_memory: bytes,
) -> str:
    memory = repository_memory.decode("utf-8")
    context_section = (
        "No additional caller context was supplied."
        if not additional_context
        else additional_context
    )
    return f"""You are the sole coding agent for one evidence-grounded ML/AI experiment.

Goal:
{goal}

Additional caller context:
{context_section}

Pinned repository memory (complete canonical JSON):
{memory}

Before choosing an idea, call `list_prior_knowledge` exactly once. Call
`get_prior_knowledge_record` for every listed record that could materially affect
the decision. Treat negative and contradictory evidence as first-class evidence.

Then inspect the repository, choose one experiment with an explicit exploration or
exploitation rationale, implement it as a coherent direct successor, and validate
the changed code using only tools available inside the workspace. Do not merely
describe edits: make them. Do not read or edit Git metadata or any path outside the
workspace. Preserve the full task constraints. Your final structured result must
truthfully report the idea, implementation, validation, used record IDs, and open
uncertainties."""


def _operation_id(
    *,
    handoff: PreparedRunHandoff,
    prompt: str,
    predecessor_source_tree_digest: str,
) -> str:
    digest = tree_or_blob_digest(
        canonical_json_bytes(
            {
                "launch_manifest_id": handoff.identity.launch_manifest_id,
                "predecessor_source_tree_digest": predecessor_source_tree_digest,
                "prompt": prompt,
                "response_schema": _EVOLUTION_RESPONSE_SCHEMA,
            }
        )
    )
    return f"agent_call_{digest[7:39]}"


def _prompt_input_digest(goal: str, additional_context: str) -> str:
    return tree_or_blob_digest(
        canonical_json_bytes(
            {
                "additional_context": additional_context,
                "goal": goal,
            }
        )
    )


def _require_pinned_prompt_inputs(
    launch_request,
    *,
    goal: str,
    additional_context: str,
) -> None:
    if launch_request.prompt_input_digest != _prompt_input_digest(
        goal,
        additional_context,
    ):
        raise ProductionEvolutionError(
            "evolution prompt inputs differ from the pinned launch"
        )


__all__ = [
    "execute_production_evolution",
    "ProductionEvolutionError",
    "ProductionEvolutionResult",
]
