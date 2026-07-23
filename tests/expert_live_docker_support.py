"""Shared failure-safe Docker lifecycle support for explicit live checks."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable


def run_setup_docker(settings, docker_config_root: Path, arguments: tuple[str, ...]):
    return subprocess.run(
        (
            settings.runtime_executable_path,
            "--host",
            f"unix://{settings.runtime_socket_path}",
            "--config",
            str(docker_config_root),
            *arguments,
        ),
        cwd=docker_config_root.parent,
        env={
            "DOCKER_API_VERSION": settings.runtime_api_version,
            "DOCKER_CONFIG": str(docker_config_root),
            "HOME": str(docker_config_root.parent),
            "LANG": "C",
            "LC_ALL": "C",
        },
        capture_output=True,
        timeout=settings.command_timeout_seconds,
        check=False,
    )


def require_setup_docker_success(
    result: subprocess.CompletedProcess,
    check_name: str,
) -> None:
    if result.returncode != 0:
        raise AssertionError(
            f"real-Docker {check_name} setup command failed:\n"
            f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
        )


def remove_exact_image(
    settings,
    docker_config_root: Path,
    image_reference: str,
) -> None:
    observed_references = _listed_resources(
        settings,
        docker_config_root,
        (
            "image",
            "ls",
            "--all",
            "--digests",
            "--no-trunc",
            "--format",
            "{{.Repository}}@{{.Digest}}",
        ),
    )
    matching_references = tuple(
        reference for reference in observed_references if reference == image_reference
    )
    if not matching_references:
        return
    if len(matching_references) != 1:
        raise AssertionError(
            "real-Docker image inventory contains a duplicate authority"
        )
    result = run_setup_docker(
        settings,
        docker_config_root,
        ("image", "rm", image_reference),
    )
    require_setup_docker_success(result, "image cleanup")


def cleanup_daemon_resources(
    settings,
    docker_config_root: Path,
    provider_handle_ids: Iterable[str],
) -> None:
    """Sweep exact live-test handles even when the provider path aborts."""

    for provider_handle_id in provider_handle_ids:
        container_ids = _listed_resources(
            settings,
            docker_config_root,
            (
                "container",
                "ls",
                "--all",
                "--quiet",
                "--filter",
                f"label=io.kapso.task-evaluation.handle={provider_handle_id}",
            ),
        )
        if container_ids:
            require_setup_docker_success(
                run_setup_docker(
                    settings,
                    docker_config_root,
                    ("container", "rm", "--force", *container_ids),
                ),
                "container cleanup",
            )
        volume_names = _listed_resources(
            settings,
            docker_config_root,
            (
                "volume",
                "ls",
                "--quiet",
                "--filter",
                f"label=io.kapso.task-evaluation.handle={provider_handle_id}",
            ),
        )
        if volume_names:
            require_setup_docker_success(
                run_setup_docker(
                    settings,
                    docker_config_root,
                    ("volume", "rm", "--force", *volume_names),
                ),
                "volume cleanup",
            )


def assert_no_daemon_resources(
    settings,
    docker_config_root: Path,
    provider_handle_ids: tuple[str, ...],
    check_name: str,
) -> None:
    for provider_handle_id in provider_handle_ids:
        for resource_kind, command in (
            (
                "container",
                (
                    "container",
                    "ls",
                    "--all",
                    "--quiet",
                    "--filter",
                    f"label=io.kapso.task-evaluation.handle={provider_handle_id}",
                ),
            ),
            (
                "volume",
                (
                    "volume",
                    "ls",
                    "--quiet",
                    "--filter",
                    f"label=io.kapso.task-evaluation.handle={provider_handle_id}",
                ),
            ),
        ):
            resources = _listed_resources(settings, docker_config_root, command)
            assert (
                resources == ()
            ), f"{check_name} leaked a handle-owned {resource_kind}: {resources!r}"


def _listed_resources(
    settings,
    docker_config_root: Path,
    command: tuple[str, ...],
) -> tuple[str, ...]:
    result = run_setup_docker(settings, docker_config_root, command)
    require_setup_docker_success(result, "resource observation")
    lines = result.stdout.splitlines()
    if any(not line or any(byte > 127 for byte in line) for line in lines):
        raise AssertionError("real-Docker resource observation is not canonical ASCII")
    return tuple(line.decode("ascii") for line in lines)
