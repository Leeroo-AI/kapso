from __future__ import annotations

import copy

import pytest

from kapso.core.config import load_config
from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.launch.run_action_contracts import (
    RunFrontierActionKind,
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_docker_projection import (
    DOCKER_RUN_ACTION_PROJECTION_PROTOCOL_VERSION,
    DOCKER_RUN_ACTION_RAW_FIELD_SCHEMA_ID,
    DockerRunActionCommand,
    DockerRunActionProjectionError,
    keeper_create_arguments,
    main_create_arguments,
    require_run_action_image,
    volume_create_arguments,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionCredentialMode,
    RunActionRuntimeVolumeAuthority,
    preparation_container_labels,
    preparation_container_name,
    preparation_keeper_container_labels,
    preparation_keeper_container_name,
    preparation_volume_labels,
    preparation_volume_name,
    run_action_docker_init_authority_id,
    run_action_supervisor_helper_authority_id,
)
from kapso.cross_run.settings import CrossRunSettings
from test_run_action_supervisor_contracts import (
    _claim,
    _execution_policy,
    _remint_policy,
    _remint_resource_limits,
    _volume_authority,
)

_CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"
_GENERATION_NONCE = "1" * 32


@pytest.fixture(scope="module")
def docker_settings():
    return CrossRunSettings.from_dict(
        load_config(_CANONICAL_CONFIG_PATH)["cross_run"]
    ).docker


def _policy(
    docker_settings,
    *,
    workspace_access=RunFrontierWorkspaceAccess.READ_ONLY,
    credential_mode=RunActionCredentialMode.SUPERVISOR_FILE,
    command_template_id=None,
):
    policy = _execution_policy(
        kind=(
            RunFrontierActionKind.EMBEDDING
            if workspace_access is RunFrontierWorkspaceAccess.NONE
            else RunFrontierActionKind.CODING_AGENT
        ),
        workspace_access=workspace_access,
        credential_mode=credential_mode,
    )
    return _remint_policy(
        policy,
        projection_protocol_version=DOCKER_RUN_ACTION_PROJECTION_PROTOCOL_VERSION,
        raw_field_schema_id=DOCKER_RUN_ACTION_RAW_FIELD_SCHEMA_ID,
        docker_runtime_settings_digest=tree_or_blob_digest(
            docker_settings.to_json_bytes()
        ),
        supervisor_helper_source_path=docker_settings.helper_executable_path,
        supervisor_helper_executable_authority_id=(
            run_action_supervisor_helper_authority_id(
                docker_settings.helper_executable_path,
                docker_settings.helper_executable_digest,
            )
        ),
        supervisor_helper_executable_digest=docker_settings.helper_executable_digest,
        docker_init_source_path=docker_settings.init_executable_path,
        docker_init_executable_authority_id=(
            run_action_docker_init_authority_id(
                docker_settings.init_executable_path,
                docker_settings.init_executable_digest,
            )
        ),
        docker_init_executable_digest=docker_settings.init_executable_digest,
        **(
            {}
            if command_template_id is None
            else {"command_template_id": command_template_id}
        ),
    )


def _claim_and_volume(docker_settings, **policy_options):
    claim = _claim(policy=_policy(docker_settings, **policy_options))
    return claim, _volume_authority(claim, nonce=_GENERATION_NONCE)


def _fixed_command(
    *,
    entrypoint="/bin/tool",
    arguments=("run",),
):
    return DockerRunActionCommand.build(
        entrypoint=entrypoint,
        arguments=arguments,
    )


def _remint_volume_authority(authority, **changes):
    values = {
        key: value
        for key, value in authority.to_dict().items()
        if key != "runtime_volume_authority_id"
    }
    values.update(changes)
    values["driver_options"] = tuple(
        sorted(
            (
                "device=tmpfs",
                (
                    "o=nodev,nosuid,noswap,"
                    f"size={values['size_limit_bytes']},"
                    f"nr_inodes={values['inode_limit']},"
                    f"mode={values['root_mode']:04o},"
                    f"uid={values['owner_user_id']},"
                    f"gid={values['owner_group_id']}"
                ),
                "type=tmpfs",
            )
        )
    )
    return RunActionRuntimeVolumeAuthority.mint(**values)


def _image(policy):
    return {
        "Architecture": policy.image_authority.architecture,
        "Comment": "",
        "Config": {
            "Cmd": ["hostile-image-command"],
            "Entrypoint": ["/bin/false"],
            "Env": ["LANG=C", "PATH=/usr/bin:/bin"],
            "ExposedPorts": None,
            "Healthcheck": None,
            "Hostname": "hostile-image-hostname",
            "Labels": None,
            "StopSignal": "SIGKILL",
            "User": "0",
            "Volumes": None,
            "WorkingDir": "/hostile-image-workdir",
        },
        "Created": "2026-07-25T00:00:00Z",
        "GraphDriver": {
            "Data": {
                "LowerDir": "/var/lib/docker/overlay2/lower",
                "MergedDir": "/var/lib/docker/overlay2/merged",
                "UpperDir": "/var/lib/docker/overlay2/upper",
                "WorkDir": "/var/lib/docker/overlay2/work",
            },
            "Name": "overlay2",
        },
        "Id": policy.image_authority.image_config_digest,
        "Metadata": {"LastTagTime": "0001-01-01T00:00:00Z"},
        "Os": policy.image_authority.operating_system,
        "RepoDigests": [policy.image_authority.image_reference],
        "RepoTags": [],
        "RootFS": {
            "Layers": ["sha256:" + "2" * 64],
            "Type": "layers",
        },
        "Size": 1024,
    }


def _label_arguments(labels):
    return tuple(
        value for label in labels for value in ("--label", f"{label.key}={label.value}")
    )


def _common_arguments(policy, working_directory):
    limits = policy.docker_resource_limits
    sandbox = policy.sandbox_spec
    return (
        "--pull",
        "never",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "apparmor:docker-default",
        "--security-opt",
        "no-new-privileges",
        "--security-opt",
        "seccomp:builtin",
        "--cgroupns",
        "private",
        "--ipc",
        "private",
        "--cgroup-parent",
        sandbox.cgroup_parent_id,
        "--runtime",
        "runc",
        "--log-driver",
        "none",
        "--init",
        "--restart",
        "no",
        "--hostname",
        policy.hostname,
        "--user",
        f"{policy.user_id}:{policy.group_id}",
        "--workdir",
        working_directory,
        "--stop-signal",
        "SIGTERM",
        "--stop-timeout",
        str(policy.supervisor_limits.termination_grace_seconds),
        "--cpu-period",
        str(limits.cpu_period_microseconds),
        "--cpu-quota",
        str(limits.cpu_quota_microseconds),
        "--cpu-shares",
        str(limits.cpu_shares),
        "--memory",
        str(limits.memory_size_bytes),
        "--memory-reservation",
        str(limits.memory_reservation_size_bytes),
        "--memory-swap",
        str(limits.memory_swap_size_bytes),
        "--oom-score-adj",
        str(limits.oom_score_adjustment),
        "--pids-limit",
        str(limits.process_limit),
        "--blkio-weight",
        str(limits.block_io_weight),
        "--shm-size",
        str(limits.shared_memory_size_bytes),
        "--env",
        "LANG=C",
        "--env",
        "PATH=/usr/bin:/bin",
    )


def test_projection_schema_identity_is_structural_and_content_addressed():
    assert (
        DOCKER_RUN_ACTION_PROJECTION_PROTOCOL_VERSION
        == "kapso.docker_run_action_create_inspect.v4"
    )
    assert DOCKER_RUN_ACTION_RAW_FIELD_SCHEMA_ID == (
        "docker-raw-field-schema:"
        "sha256:dfe5f18d8510b8910e00b8eed87386e868230162281cd7f414144a4747a4732c"
    )


def test_volume_creation_is_one_exact_canonical_tuple(docker_settings):
    claim, authority = _claim_and_volume(docker_settings)

    assert volume_create_arguments(claim, authority, docker_settings) == (
        "volume",
        "create",
        "--driver",
        "local",
        *_label_arguments(preparation_volume_labels(claim, authority.generation_nonce)),
        "--opt",
        "device=tmpfs",
        "--opt",
        (
            "o=nodev,nosuid,noswap,size=536870912,nr_inodes=4096,"
            "mode=0700,uid=1000,gid=1000"
        ),
        "--opt",
        "type=tmpfs",
        preparation_volume_name(claim),
    )


def test_keeper_creation_is_one_exact_closed_tuple(docker_settings):
    claim, authority = _claim_and_volume(docker_settings)
    policy = claim.execution_policy

    assert keeper_create_arguments(
        claim,
        authority,
        _image(policy),
        docker_settings,
    ) == (
        "container",
        "create",
        "--name",
        preparation_keeper_container_name(claim),
        *_label_arguments(preparation_keeper_container_labels(claim)),
        *_common_arguments(policy, "/kapso/runtime-volume"),
        "--mount",
        (
            "type=bind,source=/usr/bin/busybox,"
            "target=/kapso-supervisor/busybox,"
            "readonly,bind-recursive=disabled,bind-propagation=rprivate"
        ),
        "--mount",
        (
            f"type=volume,source={authority.volume_name},"
            "target=/kapso/runtime-volume,volume-nocopy"
        ),
        "--entrypoint",
        "/kapso-supervisor/busybox",
        policy.image_authority.image_reference,
        "tail",
        "-f",
        "/dev/null",
    )


def test_main_creation_uses_only_sorted_bounded_volume_subpaths(docker_settings):
    command = _fixed_command(
        entrypoint="/usr/bin/codex",
        arguments=("exec", "--json", "/kapso/input/request.blob"),
    )
    claim, authority = _claim_and_volume(
        docker_settings,
        command_template_id=command.command_template_id,
    )
    policy = claim.execution_policy

    arguments = main_create_arguments(
        claim,
        authority,
        command,
        _image(policy),
        docker_settings,
    )

    assert arguments == (
        "container",
        "create",
        "--name",
        preparation_container_name(claim),
        *_label_arguments(preparation_container_labels(claim)),
        *_common_arguments(policy, "/kapso/workspace"),
        "--mount",
        (
            "type=bind,source=/usr/bin/busybox,"
            "target=/kapso-supervisor/busybox,"
            "readonly,bind-recursive=disabled,bind-propagation=rprivate"
        ),
        "--mount",
        (
            f"type=volume,source={authority.volume_name},"
            "target=/kapso-supervisor/control,readonly,volume-nocopy,"
            "volume-subpath=control"
        ),
        "--mount",
        (
            f"type=volume,source={authority.volume_name},"
            "target=/kapso/credentials,readonly,volume-nocopy,"
            "volume-subpath=credential"
        ),
        "--mount",
        (
            f"type=volume,source={authority.volume_name},"
            "target=/kapso/input,readonly,volume-nocopy,volume-subpath=input"
        ),
        "--mount",
        (
            f"type=volume,source={authority.volume_name},"
            "target=/kapso/result,volume-nocopy,volume-subpath=result"
        ),
        "--mount",
        (
            f"type=volume,source={authority.volume_name},"
            "target=/kapso/tmp,volume-nocopy,volume-subpath=temporary"
        ),
        "--mount",
        (
            f"type=volume,source={authority.volume_name},"
            "target=/kapso/workspace,readonly,volume-nocopy,"
            "volume-subpath=workspace"
        ),
        "--entrypoint",
        "/kapso-supervisor/busybox",
        policy.image_authority.image_reference,
        "sh",
        "-eu",
        "-c",
        (
            'while [ ! -f "$1" ] || [ ! -r "$1" ]; do "$2" sleep "$3"; done; '
            'shift 3; exec "$@"'
        ),
        "kapso-run-action-barrier",
        "/kapso-supervisor/control/release",
        "/kapso-supervisor/busybox",
        str(docker_settings.run_action_barrier_poll_interval_seconds),
        "/usr/bin/codex",
        "exec",
        "--json",
        "/kapso/input/request.blob",
    )
    joined = "\n".join(arguments)
    assert ".kapso-generation" not in joined
    assert "/kapso/runtime-volume" not in joined
    assert "complete request" not in joined
    assert "credential" in joined


@pytest.mark.parametrize(
    ("workspace_access", "credential_mode", "expected_workspace_access"),
    (
        (
            RunFrontierWorkspaceAccess.NONE,
            RunActionCredentialMode.NONE,
            None,
        ),
        (
            RunFrontierWorkspaceAccess.READ_ONLY,
            RunActionCredentialMode.NONE,
            "readonly",
        ),
        (
            RunFrontierWorkspaceAccess.EDIT_WORKSPACE,
            RunActionCredentialMode.NONE,
            "read_write",
        ),
    ),
)
def test_main_mount_authority_tracks_workspace_and_credential_policy(
    docker_settings,
    workspace_access,
    credential_mode,
    expected_workspace_access,
):
    claim, authority = _claim_and_volume(
        docker_settings,
        workspace_access=workspace_access,
        credential_mode=credential_mode,
        command_template_id=_fixed_command().command_template_id,
    )
    arguments = main_create_arguments(
        claim,
        authority,
        _fixed_command(),
        _image(claim.execution_policy),
        docker_settings,
    )
    mounts = tuple(
        arguments[position + 1]
        for position, value in enumerate(arguments)
        if value == "--mount"
    )

    assert not any("credential" in mount for mount in mounts)
    workspace_mounts = tuple(
        mount for mount in mounts if "volume-subpath=workspace" in mount
    )
    if expected_workspace_access is None:
        assert workspace_mounts == ()
    elif expected_workspace_access == "readonly":
        assert len(workspace_mounts) == 1
        assert ",readonly," in workspace_mounts[0]
    else:
        assert len(workspace_mounts) == 1
        assert ",readonly," not in workspace_mounts[0]


def test_optional_cpuset_authority_is_rendered_without_defaults(docker_settings):
    policy = _policy(docker_settings)
    resource_limits = _remint_resource_limits(
        policy.docker_resource_limits,
        cpuset_cpu_ids=(0, 2),
        cpuset_memory_node_ids=(0,),
    )
    policy = _remint_policy(policy, docker_resource_limits=resource_limits)
    claim = _claim(policy=policy)
    authority = _volume_authority(claim, nonce=_GENERATION_NONCE)

    arguments = keeper_create_arguments(
        claim,
        authority,
        _image(policy),
        docker_settings,
    )

    assert arguments[arguments.index("--cpuset-cpus") + 1] == "0,2"
    assert arguments[arguments.index("--cpuset-mems") + 1] == "0"


def _add_root_field(image):
    image["Unknown"] = None


def _remove_root_field(image):
    image.pop("Size")


def _add_config_field(image):
    image["Config"]["Unknown"] = None


def _add_graph_driver_field(image):
    image["GraphDriver"]["Unknown"] = None


def _add_graph_data_field(image):
    image["GraphDriver"]["Data"]["Unknown"] = None


def _add_metadata_field(image):
    image["Metadata"]["Unknown"] = None


def _add_root_filesystem_field(image):
    image["RootFS"]["Unknown"] = None


@pytest.mark.parametrize(
    "mutate",
    (
        _add_root_field,
        _remove_root_field,
        _add_config_field,
        _add_graph_driver_field,
        _add_graph_data_field,
        _add_metadata_field,
        _add_root_filesystem_field,
    ),
)
def test_image_preflight_rejects_every_unknown_or_missing_schema_seam(
    docker_settings,
    mutate,
):
    policy = _policy(docker_settings)
    image = _image(policy)
    mutate(image)

    with pytest.raises(DockerRunActionProjectionError, match="unknown"):
        require_run_action_image(image, policy, docker_settings)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    (
        ("Volumes", {"/data": {}}, "Volumes"),
        ("ExposedPorts", {"8080/tcp": {}}, "ExposedPorts"),
        ("Labels", {"org.example.benign": "value"}, "Labels"),
        ("Healthcheck", {"Test": ["NONE"]}, "healthcheck"),
        ("Env", ["API_TOKEN=secret"], "environment"),
        ("Env", ["HOME=/root"], "environment"),
        ("Env", ["LANG=C", "LANG=C"], "environment"),
        ("Env", ["MALFORMED"], "environment"),
        ("Env", "LANG=C", "environment"),
    ),
)
def test_image_preflight_rejects_inherited_authority(
    docker_settings,
    field_name,
    value,
    message,
):
    policy = _policy(docker_settings)
    image = _image(policy)
    image["Config"][field_name] = value

    with pytest.raises(DockerRunActionProjectionError, match=message):
        require_run_action_image(image, policy, docker_settings)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    (
        ("Id", "sha256:" + "f" * 64, "content authority"),
        ("Architecture", "arm64", "content authority"),
        ("Os", "windows", "content authority"),
        ("RepoDigests", [], "content authority"),
        ("Variant", "v8", "content authority"),
        ("Variant", {"unexpected": "object"}, "variant is malformed"),
    ),
)
def test_image_preflight_rejects_content_or_platform_substitution(
    docker_settings,
    field_name,
    value,
    message,
):
    policy = _policy(docker_settings)
    image = _image(policy)
    image[field_name] = value

    with pytest.raises(DockerRunActionProjectionError, match=message):
        require_run_action_image(image, policy, docker_settings)


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("Size", {}),
        ("Size", True),
        ("RepoTags", {}),
        ("RepoTags", [""]),
        ("RepoTags", ["example/tag:latest", "example/tag:latest"]),
        ("Comment", {}),
        ("Created", {}),
        ("Created", ""),
        ("Created", "not-a-canonical-utc-timestamp"),
    ),
)
def test_image_preflight_rejects_malformed_classified_root_fields(
    docker_settings,
    field_name,
    value,
):
    policy = _policy(docker_settings)
    image = _image(policy)
    image[field_name] = value

    with pytest.raises(DockerRunActionProjectionError, match="content authority"):
        require_run_action_image(image, policy, docker_settings)


def test_image_preflight_accepts_only_exact_policy_environment_subset(
    docker_settings,
):
    policy = _policy(docker_settings)
    image = _image(policy)
    image["Config"]["Env"] = ["PATH=/usr/bin:/bin"]

    require_run_action_image(image, policy, docker_settings)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    (
        ("Name", "btrfs", "unknown or missing"),
        ("MergedDir", "/tmp/escaped", "escapes Docker"),
        (
            "LowerDir",
            "/var/lib/docker/overlay2/lower:/tmp/escaped",
            "escapes Docker",
        ),
        (
            "UpperDir",
            "/var/lib/docker/overlay2/../../../../tmp/escaped",
            "escapes Docker",
        ),
    ),
)
def test_image_preflight_rejects_graph_driver_substitution(
    docker_settings,
    field_name,
    value,
    message,
):
    policy = _policy(docker_settings)
    image = _image(policy)
    if field_name == "Name":
        image["GraphDriver"]["Name"] = value
    else:
        image["GraphDriver"]["Data"][field_name] = value

    with pytest.raises(DockerRunActionProjectionError, match=message):
        require_run_action_image(image, policy, docker_settings)


def test_renderer_rejects_policy_schema_runtime_or_helper_substitution(
    docker_settings,
):
    policy = _policy(docker_settings)
    changed_policies = (
        _remint_policy(
            policy,
            projection_protocol_version="kapso.alternate_projection.v1",
        ),
        _remint_policy(
            policy,
            raw_field_schema_id=("docker-raw-field-schema:sha256:" + "f" * 64),
        ),
        _remint_policy(
            policy,
            docker_runtime_settings_digest="sha256:" + "f" * 64,
        ),
    )
    for changed_policy in changed_policies:
        changed_claim = _claim(policy=changed_policy)
        authority = _volume_authority(changed_claim, nonce=_GENERATION_NONCE)
        with pytest.raises(DockerRunActionProjectionError, match="closed Docker"):
            keeper_create_arguments(
                changed_claim,
                authority,
                _image(changed_policy),
                docker_settings,
            )


def test_command_is_absolute_complete_and_nonempty():
    with pytest.raises(DockerRunActionProjectionError, match="absolute path"):
        DockerRunActionCommand.build(entrypoint="bin/tool", arguments=("run",))
    with pytest.raises(DockerRunActionProjectionError, match="arguments"):
        DockerRunActionCommand.build(entrypoint="/bin/tool", arguments=())
    with pytest.raises(DockerRunActionProjectionError, match="arguments"):
        DockerRunActionCommand.build(
            entrypoint="/bin/tool",
            arguments=("run", "bad\x00argument"),
        )


def test_command_identity_binds_every_persisted_argument():
    command = _fixed_command()

    with pytest.raises(DockerRunActionProjectionError, match="template identity"):
        DockerRunActionCommand(
            command_template_id=(
                "docker-run-action-command-template:sha256:" + "f" * 64
            ),
            entrypoint=command.entrypoint,
            arguments=command.arguments,
        )


def test_main_renderer_rejects_command_outside_durable_template(docker_settings):
    claim, authority = _claim_and_volume(docker_settings)
    command = _fixed_command(
        arguments=(
            "run",
            "--token=credential-canary",
            "/home/ubuntu/host-workspace",
        )
    )

    with pytest.raises(DockerRunActionProjectionError, match="execution policy"):
        main_create_arguments(
            claim,
            authority,
            command,
            _image(claim.execution_policy),
            docker_settings,
        )


def test_volume_renderer_rejects_another_claim_authority(docker_settings):
    claim, _authority = _claim_and_volume(docker_settings)
    another_claim = _claim(
        policy=_remint_policy(
            _policy(docker_settings),
            command_template_id=(
                "docker-run-action-command-template:sha256:" + "f" * 64
            ),
        ),
    )
    another_authority = _volume_authority(
        another_claim,
        nonce=_GENERATION_NONCE,
    )

    with pytest.raises(DockerRunActionProjectionError, match="preparation claim"):
        volume_create_arguments(claim, another_authority, docker_settings)


def test_volume_renderer_rejects_same_claim_policy_expansion(docker_settings):
    claim, authority = _claim_and_volume(docker_settings)
    expanded_authorities = (
        _remint_volume_authority(
            authority,
            owner_user_id=authority.owner_user_id + 1,
        ),
        _remint_volume_authority(
            authority,
            owner_group_id=authority.owner_group_id + 1,
        ),
        _remint_volume_authority(
            authority,
            size_limit_bytes=authority.size_limit_bytes * 2,
        ),
        _remint_volume_authority(
            authority,
            inode_limit=authority.inode_limit * 2,
        ),
    )

    for expanded_authority in expanded_authorities:
        with pytest.raises(DockerRunActionProjectionError, match="execution policy"):
            volume_create_arguments(claim, expanded_authority, docker_settings)


def test_rendering_does_not_mutate_image_inspection(docker_settings):
    command = _fixed_command()
    claim, authority = _claim_and_volume(
        docker_settings,
        command_template_id=command.command_template_id,
    )
    image = _image(claim.execution_policy)
    before = copy.deepcopy(image)

    keeper_create_arguments(claim, authority, image, docker_settings)
    main_create_arguments(
        claim,
        authority,
        command,
        image,
        docker_settings,
    )

    assert image == before
