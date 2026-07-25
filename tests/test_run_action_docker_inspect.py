from __future__ import annotations

import copy

import pytest

from kapso.core.config import load_config
from kapso.cross_run.launch import run_action_docker_inspect as docker_inspect
from kapso.cross_run.launch.run_action_docker_inspect import (
    DockerRunActionInspectionError,
    issued_keeper_projection,
    issued_main_projection,
    observe_inert_keeper,
    observe_inert_main_container,
    observe_running_keeper,
    observe_runtime_volume,
)
from kapso.cross_run.launch.run_action_docker_projection import (
    DockerRunActionCommand,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionKeeperHelperEvidence,
    RunActionMountedKeeperHelperEvidence,
    RunActionPreparedMountAccess,
    preparation_container_labels,
    preparation_container_name,
    preparation_keeper_container_labels,
    preparation_keeper_container_name,
    preparation_main_mounts,
    runtime_volume_driver_options,
)
from kapso.cross_run.settings import CrossRunSettings
from test_run_action_docker_projection import (
    _GENERATION_NONCE,
    _policy,
)
from test_run_action_supervisor_contracts import (
    _claim,
    _volume_authority,
)

_CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"
_MAIN_CONTAINER_ID = "a" * 64
_KEEPER_CONTAINER_ID = "b" * 64
_MAIN_STORAGE_LAYER_ID = "c" * 64
_KEEPER_STORAGE_LAYER_ID = "d" * 64
_IMAGE_STORAGE_LAYER_ID = "e" * 64


@pytest.fixture(scope="module")
def docker_settings():
    return CrossRunSettings.from_dict(
        load_config(_CANONICAL_CONFIG_PATH)["cross_run"]
    ).docker


@pytest.fixture(autouse=True)
def mounted_helper_observer(monkeypatch):
    def observe(source_evidence, *, container_id, process_id):
        return RunActionMountedKeeperHelperEvidence.mint(
            source_helper_evidence=source_evidence,
            container_id=container_id,
            process_id=process_id,
            process_cgroup_path=(
                f"/test.kapso.run_action.slice/docker-{container_id}.scope"
            ),
            destination=source_evidence.destination,
            mount_id=source_evidence.mount_id + 1,
            device=source_evidence.device,
            inode=source_evidence.inode,
            executable_digest=source_evidence.executable_digest,
        )

    monkeypatch.setattr(
        docker_inspect,
        "observe_mounted_keeper_helper",
        observe,
    )


def _context(docker_settings):
    command = DockerRunActionCommand.build(
        entrypoint="/bin/tool",
        arguments=("run", "--request", "/kapso/input/request.blob"),
    )
    policy = _policy(
        docker_settings,
        command_template_id=command.command_template_id,
    )
    claim = _claim(policy=policy)
    authority = _volume_authority(claim, nonce=_GENERATION_NONCE)
    volume_raw = _volume_raw(authority, docker_settings)
    volume = observe_runtime_volume(
        volume_raw,
        claim,
        authority,
        docker_settings,
    )
    helper = RunActionKeeperHelperEvidence.mint(
        helper_authority_id=policy.keeper_helper_executable_authority_id,
        source_path=policy.keeper_helper_source_path,
        destination="/kapso-supervisor/busybox",
        mount_type="bind",
        mount_access=RunActionPreparedMountAccess.READ_ONLY,
        recursive_bind=False,
        file_type="regular",
        owner_user_id=0,
        owner_group_id=0,
        mode=0o755,
        link_count=1,
        file_format="elf",
        dynamic_dependency_count=0,
        elf_interpreter_present=False,
        executable_digest=policy.keeper_helper_executable_digest,
        mount_id=100,
        device=200,
        inode=300,
    )
    return claim, authority, volume_raw, volume, command, helper


def _volume_raw(authority, docker_settings):
    options = {
        assignment.split("=", 1)[0]: assignment.split("=", 1)[1]
        for assignment in runtime_volume_driver_options(authority)
    }
    return {
        "CreatedAt": "2026-07-25T00:00:00Z",
        "Driver": "local",
        "Labels": {label.key: label.value for label in authority.labels},
        "Mountpoint": (
            f"{docker_settings.runtime_root_directory}/volumes/"
            f"{authority.volume_name}/_data"
        ),
        "Name": authority.volume_name,
        "Options": options,
        "Scope": "local",
    }


def _graph_driver(docker_settings, container_id, storage_layer_id):
    root = (
        f"{docker_settings.runtime_root_directory}/"
        f"{docker_settings.runtime_storage_driver}"
    )
    return {
        "Data": {
            "ID": container_id,
            "LowerDir": (
                f"{root}/{storage_layer_id}-init/diff:"
                f"{root}/{_IMAGE_STORAGE_LAYER_ID}/diff"
            ),
            "MergedDir": f"{root}/{storage_layer_id}/merged",
            "UpperDir": f"{root}/{storage_layer_id}/diff",
            "WorkDir": f"{root}/{storage_layer_id}/work",
        },
        "Name": docker_settings.runtime_storage_driver,
    }


def _none_network(*, running):
    endpoint_id = "f" * 64 if running else ""
    network_id = "1" * 64 if running else ""
    sandbox_id = "2" * 64 if running else ""
    return {
        "Networks": {
            "none": {
                "Aliases": None,
                "DNSNames": None,
                "DriverOpts": None,
                "EndpointID": endpoint_id,
                "Gateway": "",
                "GlobalIPv6Address": "",
                "GlobalIPv6PrefixLen": 0,
                "GwPriority": 0,
                "IPAMConfig": None,
                "IPAddress": "",
                "IPPrefixLen": 0,
                "IPv6Gateway": "",
                "Links": None,
                "MacAddress": "",
                "NetworkID": network_id,
            }
        },
        "Ports": {},
        "SandboxID": sandbox_id,
        "SandboxKey": (f"/var/run/docker/netns/{sandbox_id[:12]}" if running else ""),
    }


def _state(*, running):
    return {
        "Dead": False,
        "Error": "",
        "ExitCode": 0,
        "FinishedAt": "0001-01-01T00:00:00Z",
        "OOMKilled": False,
        "Paused": False,
        "Pid": 4242 if running else 0,
        "Restarting": False,
        "Running": running,
        "StartedAt": (
            "2026-07-25T00:00:01.123456789Z" if running else "0001-01-01T00:00:00Z"
        ),
        "Status": "running" if running else "created",
    }


def _container_raw(
    claim,
    authority,
    volume,
    command,
    docker_settings,
    *,
    keeper,
):
    policy = claim.execution_policy
    if keeper:
        container_id = _KEEPER_CONTAINER_ID
        storage_layer_id = _KEEPER_STORAGE_LAYER_ID
        labels = preparation_keeper_container_labels(claim)
        name = preparation_keeper_container_name(claim)
        executable = "/kapso-supervisor/busybox"
        arguments = ("tail", "-f", "/dev/null")
        working_directory = "/kapso/runtime-volume"
        host_mounts = docker_inspect._keeper_host_config_mounts(
            claim,
            authority,
        )
        top_mounts = docker_inspect._keeper_top_level_mounts(
            claim,
            authority,
            volume,
        )
    else:
        container_id = _MAIN_CONTAINER_ID
        storage_layer_id = _MAIN_STORAGE_LAYER_ID
        labels = preparation_container_labels(claim)
        name = preparation_container_name(claim)
        executable = command.entrypoint
        arguments = command.arguments
        working_directory = policy.filesystem_policy.working_directory
        mounts = preparation_main_mounts(claim, authority)
        host_mounts = docker_inspect._main_host_config_mounts(mounts)
        top_mounts = docker_inspect._main_top_level_mounts(mounts, volume)
    container_root = (
        f"{docker_settings.runtime_root_directory}/containers/{container_id}"
    )
    return {
        "AppArmorProfile": policy.sandbox_spec.apparmor_profile_id,
        "Args": list(arguments),
        "Config": docker_inspect._expected_container_config(
            claim,
            labels=labels,
            command_executable=executable,
            command_arguments=arguments,
            working_directory=working_directory,
        ),
        "Created": "2026-07-25T00:00:00.123456789Z",
        "Driver": docker_settings.runtime_storage_driver,
        "ExecIDs": None,
        "GraphDriver": _graph_driver(
            docker_settings,
            container_id,
            storage_layer_id,
        ),
        "HostConfig": docker_inspect._expected_host_config(
            claim,
            mounts=host_mounts,
            running_keeper=keeper,
        ),
        "HostnamePath": f"{container_root}/hostname" if keeper else "",
        "HostsPath": f"{container_root}/hosts" if keeper else "",
        "Id": container_id,
        "Image": policy.image_authority.image_config_digest,
        "LogPath": "",
        "MountLabel": "",
        "Mounts": top_mounts,
        "Name": f"/{name}",
        "NetworkSettings": _none_network(running=keeper),
        "Path": executable,
        "Platform": policy.image_authority.operating_system,
        "ProcessLabel": "",
        "ResolvConfPath": f"{container_root}/resolv.conf" if keeper else "",
        "RestartCount": 0,
        "State": _state(running=keeper),
    }


def _inert_keeper_raw(
    claim,
    authority,
    volume,
    command,
    docker_settings,
):
    raw = _container_raw(
        claim,
        authority,
        volume,
        command,
        docker_settings,
        keeper=True,
    )
    raw["HostConfig"]["OomKillDisable"] = False
    raw["HostnamePath"] = ""
    raw["HostsPath"] = ""
    raw["NetworkSettings"] = _none_network(running=False)
    raw["ResolvConfPath"] = ""
    raw["State"] = _state(running=False)
    return raw


def test_volume_inspection_is_closed_and_normalized(docker_settings):
    claim, authority, volume_raw, volume, _command, _helper = _context(docker_settings)

    assert volume.volume_authority_id == authority.runtime_volume_authority_id
    assert volume.volume_name == authority.volume_name
    assert volume.mountpoint == volume_raw["Mountpoint"]
    assert volume.unclassified_raw_field_count == 0
    assert volume.nonauthoritative_raw_field_count == 2
    assert (
        observe_runtime_volume(
            copy.deepcopy(volume_raw),
            claim,
            authority,
            docker_settings,
        )
        == volume
    )


def test_main_inspection_equals_issued_projection(docker_settings):
    claim, authority, _volume_raw, volume, command, _helper = _context(docker_settings)
    raw = _container_raw(
        claim,
        authority,
        volume,
        command,
        docker_settings,
        keeper=False,
    )

    evidence = observe_inert_main_container(
        raw,
        claim,
        authority,
        volume,
        command,
        docker_settings,
    )

    assert evidence.container_id == _MAIN_CONTAINER_ID
    assert evidence.issued_create_projection == issued_main_projection(
        claim,
        authority,
        command,
        docker_settings,
    )
    assert evidence.observed_inspect_projection == evidence.issued_create_projection
    assert evidence.issued_create_projection.unclassified_raw_field_count == 0


def test_keeper_inspection_equals_issued_projection(docker_settings):
    claim, authority, _volume_raw, volume, command, helper = _context(docker_settings)
    raw = _container_raw(
        claim,
        authority,
        volume,
        command,
        docker_settings,
        keeper=True,
    )

    evidence = observe_running_keeper(
        raw,
        claim,
        authority,
        volume,
        helper,
        docker_settings,
    )

    assert evidence.container_id == _KEEPER_CONTAINER_ID
    assert evidence.process_id == 4242
    assert evidence.mounted_helper_evidence.container_id == _KEEPER_CONTAINER_ID
    assert evidence.mounted_helper_evidence.process_id == 4242
    assert (
        evidence.mounted_helper_evidence.source_helper_evidence
        == evidence.issued_create_projection.helper_evidence
    )
    assert evidence.issued_create_projection == issued_keeper_projection(
        claim,
        authority,
        helper,
        docker_settings,
    )
    assert evidence.observed_inspect_projection == evidence.issued_create_projection


def test_inert_keeper_inspection_equals_issued_projection(docker_settings):
    claim, authority, _volume_raw, volume, command, helper = _context(docker_settings)
    raw = _inert_keeper_raw(
        claim,
        authority,
        volume,
        command,
        docker_settings,
    )

    observation = observe_inert_keeper(
        raw,
        claim,
        authority,
        volume,
        helper,
        docker_settings,
    )

    assert observation.container_id == _KEEPER_CONTAINER_ID
    assert observation.issued_create_projection == issued_keeper_projection(
        claim,
        authority,
        helper,
        docker_settings,
    )
    assert observation.observed_inspect_projection == (
        observation.issued_create_projection
    )


def _add_field(value):
    value["Unexpected"] = None


def _remove_field(value):
    value.pop(next(iter(value)))


@pytest.mark.parametrize(
    ("target", "mutate"),
    (
        ("root", _add_field),
        ("root", _remove_field),
        ("Config", _add_field),
        ("Config", _remove_field),
        ("HostConfig", _add_field),
        ("HostConfig", _remove_field),
        ("State", _add_field),
        ("State", _remove_field),
        ("GraphDriver", _add_field),
        ("GraphDriver", _remove_field),
        ("GraphDriver.Data", _add_field),
        ("GraphDriver.Data", _remove_field),
        ("NetworkSettings", _add_field),
        ("NetworkSettings", _remove_field),
        ("NetworkSettings.Networks.none", _add_field),
        ("NetworkSettings.Networks.none", _remove_field),
    ),
)
def test_container_inspection_rejects_unknown_or_missing_field(
    docker_settings,
    target,
    mutate,
):
    claim, authority, _volume_raw, volume, command, _helper = _context(docker_settings)
    raw = _container_raw(
        claim,
        authority,
        volume,
        command,
        docker_settings,
        keeper=False,
    )
    selected = raw if target == "root" else _nested(raw, target)
    mutate(selected)

    with pytest.raises(DockerRunActionInspectionError, match="raw field"):
        observe_inert_main_container(
            raw,
            claim,
            authority,
            volume,
            command,
            docker_settings,
        )


@pytest.mark.parametrize(
    ("target", "mutate"),
    (
        ("root", _add_field),
        ("root", _remove_field),
        ("Options", _add_field),
        ("Options", _remove_field),
    ),
)
def test_volume_inspection_rejects_unknown_or_missing_field(
    docker_settings,
    target,
    mutate,
):
    claim, authority, volume_raw, _volume, _command, _helper = _context(docker_settings)
    raw = copy.deepcopy(volume_raw)
    selected = raw if target == "root" else raw[target]
    mutate(selected)

    with pytest.raises(DockerRunActionInspectionError, match="raw field"):
        observe_runtime_volume(raw, claim, authority, docker_settings)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        ("Path", "/bin/alternate"),
        ("Args", ["run", "--token=secret"]),
        ("Config.Env", ["TOKEN=secret"]),
        ("Config.User", "0:0"),
        ("Config.Labels", {}),
        ("Config.Volumes", {"/anonymous": {}}),
        ("HostConfig.NetworkMode", "bridge"),
        ("HostConfig.ReadonlyRootfs", False),
        ("HostConfig.Privileged", True),
        ("HostConfig.SecurityOpt", []),
        ("HostConfig.Memory", 2**40),
        ("HostConfig.AutoRemove", 0),
        ("Config.AttachStderr", 1),
        ("HostConfig.Mounts.0.VolumeOptions.Subpath", "result"),
        ("Mounts.0.Name", "substituted-volume"),
        ("Mounts.0.RW", 0),
        ("State.Status", "running"),
        ("State.Pid", 99),
        ("State.Dead", 0),
        ("NetworkSettings.Networks.none.IPPrefixLen", False),
        ("NetworkSettings.Ports", {"80/tcp": []}),
        (
            "GraphDriver.Data.UpperDir",
            "/var/lib/docker/overlay2/../../tmp/escape/diff",
        ),
        ("RestartCount", 1),
    ),
)
def test_main_inspection_rejects_every_authority_expansion(
    docker_settings,
    path,
    value,
):
    claim, authority, _volume_raw, volume, command, _helper = _context(docker_settings)
    raw = _container_raw(
        claim,
        authority,
        volume,
        command,
        docker_settings,
        keeper=False,
    )
    _assign(raw, path, value)

    with pytest.raises(DockerRunActionInspectionError):
        observe_inert_main_container(
            raw,
            claim,
            authority,
            volume,
            command,
            docker_settings,
        )


@pytest.mark.parametrize(
    ("path", "value"),
    (
        ("State.Running", False),
        ("State.Pid", 0),
        ("State.StartedAt", "0001-01-01T00:00:00Z"),
        ("HostConfig.OomKillDisable", False),
        ("NetworkSettings.SandboxID", ""),
        ("NetworkSettings.Networks.none.EndpointID", ""),
        ("HostnamePath", ""),
        ("Mounts.0.RW", True),
        ("HostConfig.Mounts.0.BindOptions.NonRecursive", False),
    ),
)
def test_keeper_inspection_rejects_every_running_or_helper_expansion(
    docker_settings,
    path,
    value,
):
    claim, authority, _volume_raw, volume, command, helper = _context(docker_settings)
    raw = _container_raw(
        claim,
        authority,
        volume,
        command,
        docker_settings,
        keeper=True,
    )
    _assign(raw, path, value)

    with pytest.raises(DockerRunActionInspectionError):
        observe_running_keeper(
            raw,
            claim,
            authority,
            volume,
            helper,
            docker_settings,
        )


@pytest.mark.parametrize(
    ("path", "value"),
    (
        ("Driver", "nfs"),
        ("Scope", "global"),
        ("Labels", {}),
        ("Options.o", "size=999999999999"),
        ("Mountpoint", "/tmp/substituted"),
        ("CreatedAt", "not-a-timestamp"),
        ("CreatedAt", "2026-02-31T00:00:00Z"),
    ),
)
def test_volume_inspection_rejects_authority_substitution(
    docker_settings,
    path,
    value,
):
    claim, authority, volume_raw, _volume, _command, _helper = _context(docker_settings)
    raw = copy.deepcopy(volume_raw)
    _assign(raw, path, value)

    with pytest.raises(DockerRunActionInspectionError):
        observe_runtime_volume(raw, claim, authority, docker_settings)


def test_allowed_container_volatility_normalizes_to_one_projection(
    docker_settings,
):
    claim, authority, _volume_raw, volume, command, _helper = _context(docker_settings)
    first = _container_raw(
        claim,
        authority,
        volume,
        command,
        docker_settings,
        keeper=False,
    )
    second = copy.deepcopy(first)
    second_id = "3" * 64
    second_storage_id = "4" * 64
    second["Id"] = second_id
    second["GraphDriver"] = _graph_driver(
        docker_settings,
        second_id,
        second_storage_id,
    )
    second["Created"] = "2026-07-25T00:00:02Z"
    second["Config"]["Env"].reverse()
    second["HostConfig"]["Mounts"].reverse()
    second["Mounts"].reverse()

    first_evidence = observe_inert_main_container(
        first,
        claim,
        authority,
        volume,
        command,
        docker_settings,
    )
    second_evidence = observe_inert_main_container(
        second,
        claim,
        authority,
        volume,
        command,
        docker_settings,
    )

    assert first_evidence.container_id != second_evidence.container_id
    assert (
        first_evidence.observed_inspect_projection
        == second_evidence.observed_inspect_projection
    )


def test_command_or_volume_observation_cannot_be_spliced(docker_settings):
    claim, authority, _volume_raw, volume, command, _helper = _context(docker_settings)
    raw = _container_raw(
        claim,
        authority,
        volume,
        command,
        docker_settings,
        keeper=False,
    )
    alternate_command = DockerRunActionCommand.build(
        entrypoint="/bin/tool",
        arguments=("alternate",),
    )
    substituted_volume = type(volume)(
        volume_authority_id=volume.volume_authority_id,
        volume_name=volume.volume_name,
        mountpoint="/var/lib/docker/volumes/substituted/_data",
        created_at=volume.created_at,
        raw_field_schema_id=volume.raw_field_schema_id,
        unclassified_raw_field_count=volume.unclassified_raw_field_count,
        nonauthoritative_raw_field_count=(volume.nonauthoritative_raw_field_count),
    )

    with pytest.raises(DockerRunActionInspectionError, match="command"):
        observe_inert_main_container(
            raw,
            claim,
            authority,
            volume,
            alternate_command,
            docker_settings,
        )
    with pytest.raises(DockerRunActionInspectionError, match="volume"):
        observe_inert_main_container(
            raw,
            claim,
            authority,
            substituted_volume,
            command,
            docker_settings,
        )


@pytest.mark.parametrize("keeper", (False, True))
def test_every_container_leaf_is_classified_and_validated(
    docker_settings,
    keeper,
):
    claim, authority, _volume_raw, volume, command, helper = _context(docker_settings)
    raw = _container_raw(
        claim,
        authority,
        volume,
        command,
        docker_settings,
        keeper=keeper,
    )
    for path in _leaf_paths(raw):
        mutated = copy.deepcopy(raw)
        original = _path_value(mutated, path)
        _assign_parts(
            mutated,
            path,
            "unexpected-non-null-value" if original is None else None,
        )
        with pytest.raises(DockerRunActionInspectionError):
            if keeper:
                observe_running_keeper(
                    mutated,
                    claim,
                    authority,
                    volume,
                    helper,
                    docker_settings,
                )
            else:
                observe_inert_main_container(
                    mutated,
                    claim,
                    authority,
                    volume,
                    command,
                    docker_settings,
                )


def test_every_volume_leaf_is_classified_and_validated(docker_settings):
    claim, authority, volume_raw, _volume, _command, _helper = _context(docker_settings)
    for path in _leaf_paths(volume_raw):
        mutated = copy.deepcopy(volume_raw)
        original = _path_value(mutated, path)
        _assign_parts(
            mutated,
            path,
            "unexpected-non-null-value" if original is None else None,
        )
        with pytest.raises(DockerRunActionInspectionError):
            observe_runtime_volume(
                mutated,
                claim,
                authority,
                docker_settings,
            )


def test_every_inert_keeper_leaf_is_classified_and_validated(docker_settings):
    claim, authority, _volume_raw, volume, command, helper = _context(docker_settings)
    raw = _inert_keeper_raw(
        claim,
        authority,
        volume,
        command,
        docker_settings,
    )
    for path in _leaf_paths(raw):
        mutated = copy.deepcopy(raw)
        original = _path_value(mutated, path)
        _assign_parts(
            mutated,
            path,
            "unexpected-non-null-value" if original is None else None,
        )
        with pytest.raises(DockerRunActionInspectionError):
            observe_inert_keeper(
                mutated,
                claim,
                authority,
                volume,
                helper,
                docker_settings,
            )


def _nested(value, path):
    current = value
    for part in path.split("."):
        current = current[part]
    return current


def _assign(value, path, replacement):
    parts = path.split(".")
    current = value
    for part in parts[:-1]:
        current = current[int(part)] if part.isdigit() else current[part]
    final = parts[-1]
    if final.isdigit():
        current[int(final)] = replacement
    else:
        current[final] = replacement


def _leaf_paths(value, prefix=()):
    if isinstance(value, dict):
        return tuple(
            path
            for key, item in value.items()
            for path in _leaf_paths(item, (*prefix, key))
        )
    if isinstance(value, list):
        return tuple(
            path
            for index, item in enumerate(value)
            for path in _leaf_paths(item, (*prefix, index))
        )
    return (prefix,)


def _path_value(value, path):
    current = value
    for part in path:
        current = current[part]
    return current


def _assign_parts(value, path, replacement):
    current = value
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = replacement
