"""Domain-neutral validation for an expert repository topology and source tree."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Mapping

from kapso.cross_run.contracts import (
    ExpertModuleContract,
    ExpertRepositoryMap,
    SourceFileDescriptor,
)
from kapso.cross_run.expert.book import (
    EXPERT_BOOK_PATH,
    EXPERT_REPOSITORY_MAP_PATH,
    expert_control_paths,
)


class ExpertTopologyValidationError(ValueError):
    """An expert repository topology or its source-tree ownership is invalid."""


def validate_expert_repository_topology(
    repository_map: ExpertRepositoryMap,
    module_contracts: tuple[ExpertModuleContract, ...],
    *,
    validation_error_type: type[ValueError] = ExpertTopologyValidationError,
) -> dict[str, ExpertModuleContract]:
    """Validate the exact map-to-module graph and return modules by capability."""

    module_contract_ids = tuple(
        sorted(module.module_contract_id for module in module_contracts)
    )
    if len(module_contract_ids) != len(set(module_contract_ids)):
        raise validation_error_type(
            "candidate module contracts must be uniquely identified"
        )
    semantic_module_ids = tuple(module.module_id for module in module_contracts)
    if semantic_module_ids != tuple(sorted(set(semantic_module_ids))):
        raise validation_error_type(
            "candidate semantic module IDs must be sorted and unique"
        )
    modules = {module.module_id: module for module in module_contracts}
    nodes = {node.capability_id: node for node in repository_map.capability_nodes}
    if set(modules) != set(nodes) or any(
        nodes[module_id].module_contract_ref != modules[module_id].module_contract_id
        for module_id in modules
    ):
        raise validation_error_type(
            "candidate capability nodes and modules are not a bijection"
        )
    outgoing = {
        capability_id: tuple(
            sorted(
                edge.target_capability_id
                for edge in repository_map.dependency_edges
                if edge.source_capability_id == capability_id
            )
        )
        for capability_id in nodes
    }
    if any(
        module.dependency_capability_ids != outgoing[module.module_id]
        for module in module_contracts
    ):
        raise validation_error_type(
            "module dependencies differ from repository map edges"
        )
    for module in module_contracts:
        if not set(module.incompatible_capability_ids).issubset(modules):
            raise validation_error_type(
                "module incompatibility references an unknown capability"
            )
        for incompatible_id in module.incompatible_capability_ids:
            if module.module_id not in (
                modules[incompatible_id].incompatible_capability_ids
            ):
                raise validation_error_type("module incompatibility must be symmetric")
    return modules


def validate_expert_tree_ownership(
    repository_map: ExpertRepositoryMap,
    module_contracts: tuple[ExpertModuleContract, ...],
    source_files: Mapping[str, SourceFileDescriptor],
    *,
    validation_error_type: type[ValueError] = ExpertTopologyValidationError,
) -> None:
    """Validate generated controls, owned roots, adapter isolation, and entrypoints."""

    control_paths = set(expert_control_paths(module_contracts))
    if not control_paths.issubset(source_files):
        raise validation_error_type(
            "candidate tree omits generated expert control files"
        )
    control_root = PurePosixPath(EXPERT_REPOSITORY_MAP_PATH).parent
    book_path = PurePosixPath(EXPERT_BOOK_PATH)
    for path in source_files:
        source_path = PurePosixPath(path)
        if (
            source_path == book_path
            or source_path == control_root
            or control_root in source_path.parents
        ) and path not in control_paths:
            raise validation_error_type(
                f"candidate tree contains undeclared expert control file: {path}"
            )
    mount = PurePosixPath(repository_map.task_adapter_boundary.adapter_mount_path)
    owned_roots = {
        node.capability_id: tuple(PurePosixPath(path) for path in node.owned_paths)
        for node in repository_map.capability_nodes
    }
    for roots in owned_roots.values():
        if any(
            root == book_path
            or root == control_root
            or root in control_root.parents
            or control_root in root.parents
            for root in roots
        ):
            raise validation_error_type(
                "capability ownership overlaps generated expert controls"
            )
        if any(
            root == mount or root in mount.parents or mount in root.parents
            for root in roots
        ):
            raise validation_error_type(
                "task-adapter mount overlaps expert-owned source"
            )
    if any(
        PurePosixPath(path) == mount or mount in PurePosixPath(path).parents
        for path in source_files
    ):
        raise validation_error_type("candidate tree contains the external task adapter")
    owners_by_path: dict[str, str] = {}
    for path in source_files:
        if path in control_paths:
            continue
        source_path = PurePosixPath(path)
        owners = tuple(
            capability_id
            for capability_id, roots in owned_roots.items()
            if any(root == source_path or root in source_path.parents for root in roots)
        )
        if len(owners) != 1:
            raise validation_error_type(
                f"candidate source path needs exactly one owner: {path}"
            )
        owners_by_path[path] = owners[0]
    for capability_id, roots in owned_roots.items():
        for root in roots:
            if not any(
                owner == capability_id
                and (root == PurePosixPath(path) or root in PurePosixPath(path).parents)
                for path, owner in owners_by_path.items()
            ):
                raise validation_error_type(
                    f"candidate owned root is empty: {root.as_posix()}"
                )
    paths = set(source_files)
    module_by_id = {module.module_id: module for module in module_contracts}
    for capability_id, module in module_by_id.items():
        for path in (
            *module.entrypoint_refs,
            *module.test_refs,
            *module.replay_refs,
        ):
            if path not in paths or owners_by_path.get(path) != capability_id:
                raise validation_error_type(
                    f"module path is missing or foreign-owned: {path}"
                )
    for path in (
        *repository_map.validation_entrypoints,
        *repository_map.task_adapter_boundary.interface_entrypoint_refs,
    ):
        if path not in paths or path not in owners_by_path:
            raise validation_error_type(
                f"repository entrypoint is missing or unowned: {path}"
            )
