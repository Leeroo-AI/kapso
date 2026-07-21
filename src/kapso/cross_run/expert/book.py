"""Deterministic semantic book compilation for expert candidate trees."""

from __future__ import annotations

from kapso.cross_run.canonical import canonical_json_bytes, tree_or_blob_digest
from kapso.cross_run.contracts import (
    ExpertModuleContract,
    ExpertRepositoryMap,
    ExpertScopeContract,
)

EXPERT_BOOK_PATH = "EXPERT_REPO.md"
EXPERT_REPOSITORY_MAP_PATH = ".kapso/expert/repository-map.json"
EXPERT_MODULE_CONTRACT_ROOT = ".kapso/expert/module-contracts"


def expert_module_contract_path(module_contract_id: str) -> str:
    digest_suffix = module_contract_id.rsplit(":", 1)[1]
    return f"{EXPERT_MODULE_CONTRACT_ROOT}/{digest_suffix}.json"


def expert_control_paths(
    module_contracts: tuple[ExpertModuleContract, ...],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                EXPERT_BOOK_PATH,
                EXPERT_REPOSITORY_MAP_PATH,
                *(
                    expert_module_contract_path(module.module_contract_id)
                    for module in module_contracts
                ),
            }
        )
    )


def compile_expert_semantic_book(
    scope_contract: ExpertScopeContract,
    repository_map: ExpertRepositoryMap,
    module_contracts: tuple[ExpertModuleContract, ...],
) -> bytes:
    """Render one stable high-level guide from typed topology authority."""
    modules = {module.module_id: module for module in module_contracts}
    nodes = {node.capability_id: node for node in repository_map.capability_nodes}
    dependencies = {
        capability_id: tuple(
            sorted(
                edge.target_capability_id
                for edge in repository_map.dependency_edges
                if edge.source_capability_id == capability_id
            )
        )
        for capability_id in nodes
    }
    capability_order = _dependency_first_order(tuple(nodes), dependencies)
    lines = [
        "# Expert Repository",
        "",
        "## Purpose",
        "",
        _text(scope_contract.purpose),
        "",
        "## Non-goals",
        "",
        *_bullets(scope_contract.explicit_non_goals),
        "",
        "## Architecture invariants",
        "",
        *_bullets(repository_map.architecture_invariants),
        "",
        "## Task-adapter boundary",
        "",
        f"- External mount: {_code(repository_map.task_adapter_boundary.adapter_mount_path)}",
        "- Expert interface entrypoints:",
        *(
            f"  - {_code(path)}"
            for path in repository_map.task_adapter_boundary.interface_entrypoint_refs
        ),
        "- Inputs:",
        *(
            f"  - {_text(value)}"
            for value in repository_map.task_adapter_boundary.inputs
        ),
        "- Outputs:",
        *(
            f"  - {_text(value)}"
            for value in repository_map.task_adapter_boundary.outputs
        ),
        "- Invariants:",
        *(
            f"  - {_text(value)}"
            for value in repository_map.task_adapter_boundary.invariants
        ),
        "",
        "## Capability index",
        "",
        "| Capability | Purpose | Task families |",
        "| --- | --- | --- |",
    ]
    for capability_id in capability_order:
        module = modules[capability_id]
        node = nodes[capability_id]
        families = ", ".join(_code(value) for value in node.task_family_bindings)
        lines.append(
            f"| {_code(capability_id)} | {_text(module.purpose)} | "
            f"{families or 'shared'} |"
        )
    lines.extend(("", "## Composition flow", ""))
    for capability_id in capability_order:
        if dependencies[capability_id]:
            lines.extend(
                f"- {_code(dependency_id)} → {_code(capability_id)}"
                for dependency_id in dependencies[capability_id]
            )
        else:
            lines.append(f"- {_code(capability_id)} (independent entry)")
    lines.extend(("", "## Capability details", ""))
    for capability_id in capability_order:
        module = modules[capability_id]
        node = nodes[capability_id]
        lines.extend(
            (
                f"### {_text(capability_id)}",
                "",
                _text(module.purpose),
                "",
                f"- Contract version: {_code(module.version)}",
                "- Problem signals:",
                *(f"  - {_text(value)}" for value in module.problem_signals),
                "- Owned paths:",
                *(f"  - {_code(path)}" for path in node.owned_paths),
                "- Entrypoints:",
                *(f"  - {_code(path)}" for path in module.entrypoint_refs),
                "- Tests:",
                *(f"  - {_code(path)}" for path in module.test_refs),
                "- Replay references:",
                *(
                    (f"  - {_code(path)}" for path in module.replay_refs)
                    if module.replay_refs
                    else ("  - None",)
                ),
                "- Inputs:",
                *(
                    (f"  - {_text(value)}" for value in module.inputs)
                    if module.inputs
                    else ("  - None",)
                ),
                "- Outputs:",
                *(f"  - {_text(value)}" for value in module.outputs),
                "- Preconditions:",
                *(
                    (f"  - {_text(value)}" for value in module.preconditions)
                    if module.preconditions
                    else ("  - None",)
                ),
                "- Human-readable incompatibilities:",
                *(
                    (f"  - {_text(value)}" for value in module.incompatibilities)
                    if module.incompatibilities
                    else ("  - None",)
                ),
                "- Capability dependencies:",
                *(
                    (f"  - {_code(value)}" for value in dependencies[capability_id])
                    if dependencies[capability_id]
                    else ("  - None",)
                ),
                "- Incompatible capabilities:",
                *(
                    (
                        f"  - {_code(value)}"
                        for value in module.incompatible_capability_ids
                    )
                    if module.incompatible_capability_ids
                    else ("  - None",)
                ),
                "- Resource bounds:",
                "    "
                + _text(canonical_json_bytes(module.resource_bounds).decode("utf-8")),
                "- Dependency and license manifest:",
                "    "
                + _text(
                    canonical_json_bytes(module.dependency_license_manifest).decode(
                        "utf-8"
                    )
                ),
                "- Supporting episode IDs:",
                *(
                    (f"  - {_code(value)}" for value in module.supporting_episode_ids)
                    if module.supporting_episode_ids
                    else ("  - None",)
                ),
                "- Known failure episode IDs:",
                *(
                    (
                        f"  - {_code(value)}"
                        for value in module.known_failure_episode_ids
                    )
                    if module.known_failure_episode_ids
                    else ("  - None",)
                ),
                "",
            )
        )
    lines.extend(("## Validation entrypoints", ""))
    lines.extend(f"- {_code(path)}" for path in repository_map.validation_entrypoints)
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def expert_semantic_book_digest(book: bytes) -> str:
    return tree_or_blob_digest(book)


def _dependency_first_order(
    capability_ids: tuple[str, ...],
    dependencies: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    ordered: list[str] = []
    visited: set[str] = set()

    def visit(capability_id: str) -> None:
        if capability_id in visited:
            return
        for dependency_id in dependencies[capability_id]:
            visit(dependency_id)
        visited.add(capability_id)
        ordered.append(capability_id)

    for capability_id in sorted(capability_ids):
        visit(capability_id)
    return tuple(ordered)


def _bullets(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"- {_text(value)}" for value in values)


def _code(value: str) -> str:
    escaped = (
        value.replace("\r", " ")
        .replace("\n", " ")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("|", "\\|")
        .replace("`", "&#96;")
    )
    return f"`{escaped}`"


def _text(value: str) -> str:
    return (
        value.replace("\r", " ")
        .replace("\n", " ")
        .replace("\\", "\\\\")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace("|", "\\|")
        .replace("`", "\\`")
        .replace("*", "\\*")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )
