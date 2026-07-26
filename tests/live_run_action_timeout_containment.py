"""Explicit real-Docker validation for the closed timeout signal authority.

Run directly:

    pytest -q tests/live_run_action_timeout_containment.py -s
"""

from __future__ import annotations

import re
import subprocess
from contextlib import ExitStack
from pathlib import Path

from expert_live_docker_support import (
    remove_exact_image,
    require_setup_docker_success,
    run_setup_docker,
)
from kapso.core.config import load_config
from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.docker.runtime import (
    _DOCKER_CONTAINMENT_SIGNAL_AUTHORITY,
    PinnedDockerRuntime,
    read_verified_root_executable,
)
from kapso.cross_run.launch.run_action_containment_contracts import (
    RunActionTimeoutContainmentSignal,
)
from kapso.cross_run.process import BoundedProcessOutcome
from kapso.cross_run.settings import CrossRunSettings
from live_expert_replay_docker import _start_local_oci_registry

_CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"
_CONTAINER_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ORIGINAL_SUBPROCESS_RUN = subprocess.run


def test_real_docker_timeout_authority_dispatches_exact_term_and_kill(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(subprocess, "run", _ORIGINAL_SUBPROCESS_RUN)
    tmp_path.chmod(0o700)
    settings = CrossRunSettings.from_dict(
        load_config(_CANONICAL_CONFIG_PATH)["cross_run"]
    ).docker
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
        pull = run_setup_docker(
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
        require_setup_docker_success(pull, "timeout containment image")

        runtime_root = tmp_path / "runtime"
        runtime_root.mkdir(mode=0o700)
        runtime = PinnedDockerRuntime.create(
            trusted_root=runtime_root.resolve(),
            settings=settings,
        )
        containment_authority = runtime.issue_containment_authority()
        test_identity = tree_or_blob_digest(str(tmp_path).encode()).removeprefix(
            "sha256:"
        )

        for signal, expected_exit_code in (
            (RunActionTimeoutContainmentSignal.TERMINATE, 143),
            (RunActionTimeoutContainmentSignal.KILL, 137),
        ):
            container_name = (
                f"kapso-live-timeout-{signal.value.lower()}-{test_identity}"
            )
            with ExitStack() as container_cleanup:
                created = runtime.run_control(
                    (
                        "container",
                        "create",
                        "--name",
                        container_name,
                        "--init",
                        "--network",
                        "none",
                        "--read-only",
                        "--cap-drop",
                        "ALL",
                        "--security-opt",
                        "no-new-privileges",
                        local_registry.image_reference,
                        "/bin/busybox",
                        "sleep",
                        str(2 * settings.command_timeout_seconds),
                    )
                )
                container_id = created.stdout.decode("ascii").strip()
                assert _CONTAINER_ID_PATTERN.fullmatch(container_id) is not None
                container_cleanup.callback(
                    runtime.run_control,
                    ("container", "rm", "--force", "--volumes", container_id),
                )
                started = runtime.run_control(("container", "start", container_id))
                assert started.stdout == f"{container_id}\n".encode("ascii")
                running = runtime.run_json_control(
                    (
                        "container",
                        "inspect",
                        "--format",
                        "{{json .}}",
                        container_id,
                    )
                )
                assert running["State"]["Running"] is True

                dispatched = containment_authority._signal_container_once(
                    container_id=container_id,
                    signal_name=signal.value,
                    _authority=_DOCKER_CONTAINMENT_SIGNAL_AUTHORITY,
                )
                assert dispatched.outcome is BoundedProcessOutcome.COMPLETED
                assert dispatched.returncode == 0
                assert dispatched.stdout == f"{container_id}\n".encode("ascii")
                assert dispatched.stderr == b""
                waited = runtime.run_control(("container", "wait", container_id))
                assert waited.stdout == f"{expected_exit_code}\n".encode("ascii")
                terminal = runtime.run_json_control(
                    (
                        "container",
                        "inspect",
                        "--format",
                        "{{json .}}",
                        container_id,
                    )
                )
                assert terminal["Id"] == container_id
                assert terminal["State"]["Running"] is False
                assert terminal["State"]["OOMKilled"] is False
                assert terminal["State"]["ExitCode"] == expected_exit_code

            absent = runtime.run_control(
                (
                    "container",
                    "ls",
                    "--all",
                    "--no-trunc",
                    "--quiet",
                    "--filter",
                    f"name=^/{container_name}$",
                )
            )
            assert absent.stdout == b""

        assert local_registry.server.observed_violations == ()
        assert tree_or_blob_digest(busybox_bytes) == settings.helper_executable_digest
