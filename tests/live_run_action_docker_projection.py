"""Explicit real-Docker validation for issued run-action create projections.

Run directly:

    pytest -q tests/live_run_action_docker_projection.py -s
"""

from __future__ import annotations

import json
import re
from contextlib import ExitStack
from pathlib import Path

import pytest

from expert_live_docker_support import (
    remove_exact_image,
    require_setup_docker_success,
    run_setup_docker,
)
from kapso.core.config import load_config
from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.docker.runtime import (
    DockerImageAuthority,
    PinnedDockerRuntime,
    read_verified_root_executable,
)
from kapso.cross_run.launch.run_action_docker_projection import (
    DockerRunActionCommand,
    keeper_create_arguments,
    main_create_arguments,
    require_run_action_image,
    volume_create_arguments,
)
from kapso.cross_run.launch.run_action_contracts import (
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_docker_resources import (
    DockerRunActionResourceManager,
)
from kapso.cross_run.launch.run_action_docker_inspect import (
    observe_inert_keeper,
    observe_inert_main_container,
    observe_running_keeper,
    observe_runtime_volume,
)
from kapso.cross_run.launch.run_action_keeper_helper import (
    RunActionKeeperHelperError,
    observe_keeper_helper,
)
from kapso.cross_run.launch.run_action_runtime_volume import (
    RunActionRuntimeVolumeError,
    materialize_runtime_volume_layout,
    observe_empty_runtime_volume,
    reobserve_runtime_volume_layout,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionCredentialMode,
    RunActionPreparationAllocation,
    RunActionPreparedExecution,
    RunActionStaticEnvironmentVariable,
    preparation_container_labels,
    preparation_container_name,
    preparation_keeper_container_labels,
    preparation_keeper_container_name,
    preparation_volume_labels,
    preparation_volume_name,
)
from kapso.cross_run.settings import CrossRunSettings
from live_expert_replay_docker import _start_local_oci_registry
from test_run_action_docker_projection import (
    _GENERATION_NONCE,
    _policy,
)
from test_run_action_supervisor_contracts import (
    _claim,
    _remint_contract,
    _remint_policy,
    _volume_authority,
)

_CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"
_CONTAINER_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _remove_owned_container(
    settings,
    docker_config_root: Path,
    container_name: str,
    labels,
) -> None:
    arguments = [
        "container",
        "ls",
        "--all",
        "--no-trunc",
        "--quiet",
        "--filter",
        f"name=^/{container_name}$",
    ]
    for label in labels:
        arguments.extend(("--filter", f"label={label.key}={label.value}"))
    observation = run_setup_docker(
        settings,
        docker_config_root,
        tuple(arguments),
    )
    require_setup_docker_success(observation, "run-action container cleanup lookup")
    if observation.stdout == b"":
        return
    container_ids = observation.stdout.decode("ascii").splitlines()
    if (
        len(container_ids) != 1
        or _CONTAINER_ID_PATTERN.fullmatch(container_ids[0]) is None
    ):
        raise AssertionError("run-action cleanup container lookup was ambiguous")
    container_id = container_ids[0]
    label_observation = run_setup_docker(
        settings,
        docker_config_root,
        (
            "container",
            "inspect",
            "--format",
            "{{json .Config.Labels}}",
            container_id,
        ),
    )
    require_setup_docker_success(
        label_observation,
        "run-action container cleanup label inspection",
    )
    if json.loads(label_observation.stdout) != {
        label.key: label.value for label in labels
    }:
        raise AssertionError("run-action cleanup container labels differ")
    removal = run_setup_docker(
        settings,
        docker_config_root,
        ("container", "rm", "--force", "--volumes", container_id),
    )
    require_setup_docker_success(removal, "run-action container cleanup")


def _remove_owned_volume(
    settings,
    docker_config_root: Path,
    volume_name: str,
    labels,
) -> None:
    arguments = [
        "volume",
        "ls",
        "--quiet",
        "--filter",
        f"name=^{volume_name}$",
    ]
    for label in labels:
        arguments.extend(("--filter", f"label={label.key}={label.value}"))
    result = run_setup_docker(settings, docker_config_root, tuple(arguments))
    require_setup_docker_success(result, "run-action volume cleanup lookup")
    if result.stdout == b"":
        return
    if result.stdout != f"{volume_name}\n".encode("ascii"):
        raise AssertionError("run-action cleanup volume lookup was ambiguous")
    label_observation = run_setup_docker(
        settings,
        docker_config_root,
        (
            "volume",
            "inspect",
            "--format",
            "{{json .Labels}}",
            volume_name,
        ),
    )
    require_setup_docker_success(
        label_observation,
        "run-action volume cleanup label inspection",
    )
    if json.loads(label_observation.stdout) != {
        label.key: label.value for label in labels
    }:
        raise AssertionError("run-action cleanup volume labels differ")
    removal = run_setup_docker(
        settings,
        docker_config_root,
        ("volume", "rm", "--force", volume_name),
    )
    require_setup_docker_success(removal, "run-action volume cleanup")


def _listed_exact(
    settings,
    docker_config_root: Path,
    arguments: tuple[str, ...],
) -> tuple[str, ...]:
    result = run_setup_docker(settings, docker_config_root, arguments)
    require_setup_docker_success(result, "run-action projection inventory")
    return tuple(line.decode("ascii") for line in result.stdout.splitlines())


def test_real_docker_accepts_only_the_issued_run_action_projection(
    tmp_path: Path,
) -> None:
    tmp_path.chmod(0o700)
    cross_run_settings = CrossRunSettings.from_dict(
        load_config(_CANONICAL_CONFIG_PATH)["cross_run"]
    )
    settings = cross_run_settings.docker
    busybox_bytes = read_verified_root_executable(
        Path(settings.helper_executable_path),
        settings.helper_executable_digest,
    )

    with ExitStack() as cleanup:
        local_registry = _start_local_oci_registry(cleanup, busybox_bytes)
        docker_config_root = tmp_path / "setup-docker-config"
        docker_config_root.mkdir(mode=0o700)
        docker_config_path = docker_config_root / "config.json"
        docker_config_path.write_bytes(b'{"auths":{}}\n')
        docker_config_path.chmod(0o400)
        cleanup.callback(
            remove_exact_image,
            settings,
            docker_config_root,
            local_registry.image_reference,
        )
        pull_result = run_setup_docker(
            settings,
            docker_config_root,
            (
                "image",
                "pull",
                "--platform",
                "linux/amd64",
                local_registry.image_reference,
            ),
        )
        require_setup_docker_success(pull_result, "run-action projection image")
        assert local_registry.server.observed_violations == ()

        runtime_root = tmp_path / "runtime"
        runtime_root.mkdir(mode=0o700)
        runtime = PinnedDockerRuntime.create(
            trusted_root=runtime_root.resolve(),
            settings=settings,
        )
        resource_manager = DockerRunActionResourceManager(runtime)
        image_authority = DockerImageAuthority.mint(
            image_reference=local_registry.image_reference,
            image_config_digest=local_registry.config_digest,
            operating_system="linux",
            architecture="amd64",
            architecture_variant=None,
        )
        command = DockerRunActionCommand.build(
            entrypoint="/bin/busybox",
            arguments=("true",),
        )
        policy = _remint_policy(
            _policy(settings),
            image_authority=image_authority,
            command_template_id=command.command_template_id,
            static_environment=(
                RunActionStaticEnvironmentVariable(key="LANG", value="C"),
                RunActionStaticEnvironmentVariable(key="PATH", value="/bin"),
            ),
        )
        helper_evidence = observe_keeper_helper(policy)
        claim = _claim(policy=policy)
        authority = _volume_authority(claim, nonce=_GENERATION_NONCE)
        allocation = RunActionPreparationAllocation.mint(
            preparation_claim=claim,
            runtime_volume_authority=authority,
        )
        claim = allocation.preparation_claim
        authority = allocation.runtime_volume_authority
        main_name = preparation_container_name(claim)
        main_labels = preparation_container_labels(claim)
        keeper_name = preparation_keeper_container_name(claim)
        keeper_labels = preparation_keeper_container_labels(claim)
        volume_name = preparation_volume_name(claim)
        volume_labels = preparation_volume_labels(
            claim,
            authority.generation_nonce,
        )
        for name in (main_name, keeper_name):
            assert (
                _listed_exact(
                    settings,
                    docker_config_root,
                    (
                        "container",
                        "ls",
                        "--all",
                        "--quiet",
                        "--filter",
                        f"name=^/{name}$",
                    ),
                )
                == ()
            )
        assert (
            _listed_exact(
                settings,
                docker_config_root,
                (
                    "volume",
                    "ls",
                    "--quiet",
                    "--filter",
                    f"name=^{volume_name}$",
                ),
            )
            == ()
        )
        assert resource_manager.observe(allocation).is_absent

        image = runtime.inspect_exact_image(image_authority)
        require_run_action_image(image, policy, settings)

        cleanup.callback(
            _remove_owned_volume,
            settings,
            docker_config_root,
            volume_name,
            volume_labels,
        )
        volume_result = runtime.run_control(
            volume_create_arguments(claim, authority, settings)
        )
        assert volume_result.stdout == f"{volume_name}\n".encode("ascii")
        volume_inventory = resource_manager.observe(allocation)
        assert volume_inventory.volume_present is True
        assert volume_inventory.keeper_container_id is None
        assert volume_inventory.main_container_id is None
        volume_observation = observe_runtime_volume(
            resource_manager.inspect_volume(volume_inventory),
            claim,
            authority,
            settings,
        )
        cleanup.callback(
            _remove_owned_container,
            settings,
            docker_config_root,
            keeper_name,
            keeper_labels,
        )
        keeper_result = runtime.run_control(
            keeper_create_arguments(
                claim,
                authority,
                image,
                settings,
            )
        )
        keeper_id = keeper_result.stdout.decode("ascii").strip()
        assert _CONTAINER_ID_PATTERN.fullmatch(keeper_id) is not None
        inert_keeper_inventory = resource_manager.observe(allocation)
        assert inert_keeper_inventory.volume_present is True
        assert inert_keeper_inventory.keeper_container_id == keeper_id
        assert inert_keeper_inventory.main_container_id is None
        inert_keeper = observe_inert_keeper(
            resource_manager.inspect_keeper(inert_keeper_inventory),
            claim,
            authority,
            volume_observation,
            helper_evidence,
            settings,
        )
        assert inert_keeper.container_id == keeper_id
        started_keeper = runtime.run_control(("container", "start", keeper_id))
        assert started_keeper.stdout == f"{keeper_id}\n".encode("ascii")
        empty_volume_inventory = resource_manager.observe(allocation)
        assert empty_volume_inventory.volume_present is True
        assert empty_volume_inventory.keeper_container_id == keeper_id
        assert empty_volume_inventory.main_container_id is None
        empty_volume_keeper = observe_running_keeper(
            resource_manager.inspect_keeper(empty_volume_inventory),
            claim,
            authority,
            volume_observation,
            helper_evidence,
            settings,
        )
        empty_volume = observe_empty_runtime_volume(
            authority,
            volume_observation,
            empty_volume_keeper,
        )
        assert empty_volume.keeper_container_id == keeper_id
        assert empty_volume.filesystem_type == "tmpfs"
        assert empty_volume.observed_mount_flags == (
            "nodev",
            "nosuid",
            "noswap",
        )
        assert empty_volume.empty_entry_count == 0
        assert empty_volume.empty_size_bytes == 0
        assert (
            empty_volume.used_size_bytes + empty_volume.available_size_bytes
            == empty_volume.effective_size_bytes
        )
        assert (
            empty_volume.used_inode_count + empty_volume.available_inode_count
            == empty_volume.effective_inode_limit
        )
        runtime.run_control(
            (
                "container",
                "exec",
                keeper_id,
                "/kapso-supervisor/busybox",
                "mkdir",
                "-p",
                "/kapso/runtime-volume/credential",
                "/kapso/runtime-volume/input",
                "/kapso/runtime-volume/result",
                "/kapso/runtime-volume/temporary",
                "/kapso/runtime-volume/workspace",
            )
        )
        keeper_inventory = resource_manager.observe(allocation)
        assert keeper_inventory.volume_present is True
        assert keeper_inventory.keeper_container_id == keeper_id
        assert keeper_inventory.main_container_id is None

        cleanup.callback(
            _remove_owned_container,
            settings,
            docker_config_root,
            main_name,
            main_labels,
        )
        main_result = runtime.run_control(
            main_create_arguments(
                claim,
                authority,
                command,
                image,
                settings,
            )
        )
        main_id = main_result.stdout.decode("ascii").strip()
        assert _CONTAINER_ID_PATTERN.fullmatch(main_id) is not None

        complete_inventory = resource_manager.observe(allocation)
        assert complete_inventory.volume_present is True
        assert complete_inventory.keeper_container_id == keeper_id
        assert complete_inventory.main_container_id == main_id
        keeper = resource_manager.inspect_keeper(complete_inventory)
        main = resource_manager.inspect_main(complete_inventory)
        main_evidence = observe_inert_main_container(
            main,
            claim,
            authority,
            volume_observation,
            command,
            settings,
        )
        assert main_evidence.container_id == main_id
        assert (
            main_evidence.observed_inspect_projection
            == main_evidence.issued_create_projection
        )
        keeper_evidence = observe_running_keeper(
            keeper,
            claim,
            authority,
            volume_observation,
            helper_evidence,
            settings,
        )
        assert keeper_evidence.container_id == keeper_id
        assert (
            keeper_evidence.observed_inspect_projection
            == keeper_evidence.issued_create_projection
        )
        substituted_helper_evidence = _remint_contract(
            helper_evidence,
            mount_id=helper_evidence.mount_id + 1,
            device=helper_evidence.device + 1,
            inode=helper_evidence.inode + 1,
        )
        with pytest.raises(
            RunActionKeeperHelperError,
            match="differs from its issued source inode",
        ):
            observe_running_keeper(
                keeper,
                claim,
                authority,
                volume_observation,
                substituted_helper_evidence,
                settings,
            )
        assert keeper["State"]["Status"] == "running"
        assert keeper["State"]["Pid"] > 0
        assert keeper["HostConfig"]["NetworkMode"] == "none"
        assert len(keeper["Mounts"]) == 2
        assert main["State"]["Status"] == "created"
        assert main["State"]["Pid"] == 0
        assert main["RestartCount"] == 0
        assert main["Path"] == "/bin/busybox"
        assert main["Args"] == ["true"]
        host_mounts = main["HostConfig"]["Mounts"]
        assert len(host_mounts) == 5
        assert {mount["VolumeOptions"]["Subpath"] for mount in host_mounts} == {
            "credential",
            "input",
            "result",
            "temporary",
            "workspace",
        }
        assert all("Subpath" not in mount for mount in main["Mounts"])
        assert len(main["Mounts"]) == 5

        runtime.run_control(("container", "rm", "--force", "--volumes", main_id))
        runtime.run_control(("container", "rm", "--force", "--volumes", keeper_id))
        runtime.run_control(("volume", "rm", volume_name))

        assert (
            _listed_exact(
                settings,
                docker_config_root,
                (
                    "container",
                    "ls",
                    "--all",
                    "--quiet",
                    "--filter",
                    f"name=^/{main_name}$",
                ),
            )
            == ()
        )
        assert (
            _listed_exact(
                settings,
                docker_config_root,
                (
                    "container",
                    "ls",
                    "--all",
                    "--quiet",
                    "--filter",
                    f"name=^/{keeper_name}$",
                ),
            )
            == ()
        )
        assert (
            _listed_exact(
                settings,
                docker_config_root,
                (
                    "volume",
                    "ls",
                    "--quiet",
                    "--filter",
                    f"name=^{volume_name}$",
                ),
            )
            == ()
        )

        layout_policy = _remint_policy(
            _policy(
                settings,
                workspace_access=RunFrontierWorkspaceAccess.NONE,
                credential_mode=RunActionCredentialMode.NONE,
            ),
            image_authority=image_authority,
            command_template_id=command.command_template_id,
            static_environment=(
                RunActionStaticEnvironmentVariable(key="LANG", value="C"),
                RunActionStaticEnvironmentVariable(key="PATH", value="/bin"),
            ),
        )
        layout_claim = _claim(policy=layout_policy)
        layout_authority = _volume_authority(
            layout_claim,
            nonce="b" * 32,
        )
        layout_allocation = RunActionPreparationAllocation.mint(
            preparation_claim=layout_claim,
            runtime_volume_authority=layout_authority,
        )
        layout_claim = layout_allocation.preparation_claim
        layout_authority = layout_allocation.runtime_volume_authority
        layout_volume_name = preparation_volume_name(layout_claim)
        layout_volume_labels = preparation_volume_labels(
            layout_claim,
            layout_authority.generation_nonce,
        )
        layout_keeper_name = preparation_keeper_container_name(layout_claim)
        layout_keeper_labels = preparation_keeper_container_labels(layout_claim)
        layout_main_name = preparation_container_name(layout_claim)
        layout_main_labels = preparation_container_labels(layout_claim)
        cleanup.callback(
            _remove_owned_volume,
            settings,
            docker_config_root,
            layout_volume_name,
            layout_volume_labels,
        )
        layout_volume_result = runtime.run_control(
            volume_create_arguments(
                layout_claim,
                layout_authority,
                settings,
            )
        )
        assert layout_volume_result.stdout == (
            f"{layout_volume_name}\n".encode("ascii")
        )
        layout_volume_inventory = resource_manager.observe(layout_allocation)
        layout_volume_observation = observe_runtime_volume(
            resource_manager.inspect_volume(layout_volume_inventory),
            layout_claim,
            layout_authority,
            settings,
        )
        cleanup.callback(
            _remove_owned_container,
            settings,
            docker_config_root,
            layout_keeper_name,
            layout_keeper_labels,
        )
        layout_keeper_result = runtime.run_control(
            keeper_create_arguments(
                layout_claim,
                layout_authority,
                image,
                settings,
            )
        )
        layout_keeper_id = layout_keeper_result.stdout.decode("ascii").strip()
        runtime.run_control(("container", "start", layout_keeper_id))
        layout_keeper_inventory = resource_manager.observe(layout_allocation)
        layout_keeper_evidence = observe_running_keeper(
            resource_manager.inspect_keeper(layout_keeper_inventory),
            layout_claim,
            layout_authority,
            layout_volume_observation,
            helper_evidence,
            settings,
        )
        layout_empty_volume = observe_empty_runtime_volume(
            layout_authority,
            layout_volume_observation,
            layout_keeper_evidence,
        )
        prepared_volume = materialize_runtime_volume_layout(
            layout_claim,
            layout_empty_volume,
            layout_keeper_evidence,
            workspace_descriptor=None,
            settings=cross_run_settings.launch,
        )
        cleanup.callback(
            _remove_owned_container,
            settings,
            docker_config_root,
            layout_main_name,
            layout_main_labels,
        )
        layout_main_result = runtime.run_control(
            main_create_arguments(
                layout_claim,
                layout_authority,
                command,
                image,
                settings,
            )
        )
        layout_main_id = layout_main_result.stdout.decode("ascii").strip()
        layout_complete_inventory = resource_manager.observe(layout_allocation)
        layout_main_evidence = observe_inert_main_container(
            resource_manager.inspect_main(layout_complete_inventory),
            layout_claim,
            layout_authority,
            layout_volume_observation,
            command,
            settings,
        )
        prepared_execution = RunActionPreparedExecution.mint(
            preparation_claim=layout_claim,
            runtime_volume_authority=layout_authority,
            runtime_volume_evidence=prepared_volume.runtime_volume_evidence,
            volume_keeper_evidence=layout_keeper_evidence,
            input_delivery_slot=prepared_volume.input_delivery_slot,
            result_file=prepared_volume.result_file,
            credential_delivery_slot=prepared_volume.credential_delivery_slot,
            workspace_proof=prepared_volume.workspace_proof,
            layout_proof=prepared_volume.layout_proof,
            inert_container_evidence=layout_main_evidence,
        )
        reopened_volume = reobserve_runtime_volume_layout(
            prepared_execution,
            layout_volume_observation,
            layout_keeper_evidence,
            settings=cross_run_settings.launch,
        )
        assert reopened_volume == prepared_volume
        assert prepared_volume.credential_delivery_slot is None
        assert prepared_volume.workspace_proof is None
        assert prepared_volume.runtime_volume_evidence.root_inode == (
            layout_empty_volume.root_inode
        )
        assert prepared_volume.runtime_volume_evidence.sentinel_evidence.inode != (
            layout_empty_volume.root_inode
        )
        runtime.run_control(
            (
                "container",
                "exec",
                layout_keeper_id,
                "/kapso-supervisor/busybox",
                "touch",
                "/kapso/runtime-volume/unexpected",
            )
        )
        with pytest.raises(
            RunActionRuntimeVolumeError,
            match="root topology is incomplete",
        ):
            reobserve_runtime_volume_layout(
                prepared_execution,
                layout_volume_observation,
                layout_keeper_evidence,
                settings=cross_run_settings.launch,
            )
        runtime.run_control(
            (
                "container",
                "exec",
                layout_keeper_id,
                "/kapso-supervisor/busybox",
                "rm",
                "/kapso/runtime-volume/unexpected",
            )
        )
        runtime.run_control(
            (
                "container",
                "exec",
                layout_keeper_id,
                "/kapso-supervisor/busybox",
                "chmod",
                "600",
                "/kapso/runtime-volume/.kapso-generation",
            )
        )
        with pytest.raises(
            RunActionRuntimeVolumeError,
            match="file is unsafe or substituted",
        ):
            reobserve_runtime_volume_layout(
                prepared_execution,
                layout_volume_observation,
                layout_keeper_evidence,
                settings=cross_run_settings.launch,
            )

        runtime.run_control(("container", "rm", "--force", "--volumes", layout_main_id))
        runtime.run_control(
            ("container", "rm", "--force", "--volumes", layout_keeper_id)
        )
        runtime.run_control(("volume", "rm", layout_volume_name))
        assert tree_or_blob_digest(busybox_bytes) == settings.helper_executable_digest
