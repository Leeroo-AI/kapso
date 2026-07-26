from __future__ import annotations

import json
from dataclasses import replace

import pytest

import kapso.cross_run.docker.runtime as runtime_module
import kapso.cross_run.launch.run_action_docker_resources as resources_module
from kapso.core.config import load_config
from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.docker.runtime import (
    PinnedDockerRuntime,
    PinnedDockerRuntimeError,
)
from kapso.cross_run.launch.run_action_docker_resources import (
    DockerRunActionResourceError,
    DockerRunActionResourceManager,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionPreparationAllocation,
    issue_runtime_volume_authority,
    preparation_container_labels,
    preparation_container_name,
    preparation_keeper_container_labels,
    preparation_keeper_container_name,
    preparation_volume_labels,
    preparation_volume_name,
)
from kapso.cross_run.process import BoundedProcessOutcome, BoundedProcessResult
from kapso.cross_run.settings import CrossRunSettings
from test_cross_run_docker_runtime import _info, _version
from test_run_action_supervisor_contracts import _claim

_CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"
_TEST_DOCKER_BYTES = b"run-action resource inventory Docker"
_KEEPER_CONTAINER_ID = "a" * 64
_MAIN_CONTAINER_ID = "b" * 64


class _InventoryDockerRunner:
    def __init__(self, settings):
        self.settings = settings
        self.containers = {}
        self.volumes = {}
        self.requests = []
        self.lookup_count = 0
        self.mutate_after_lookup_count = None
        self.lookup_mutation = None
        self.next_lookup_stdout = None

    def run(self, request):
        self.requests.append(request)
        arguments = request.argv[5:]
        stdout = self._stdout(arguments)
        return BoundedProcessResult(
            request=request,
            outcome=BoundedProcessOutcome.COMPLETED,
            returncode=0,
            stdout=stdout,
            stderr=b"",
            stdout_bytes_observed=len(stdout),
            stderr_bytes_observed=0,
            duration_seconds=0.0,
        )

    def _stdout(self, arguments):
        if arguments == ("version", "--format", "{{json .}}"):
            return _json_line(_version(self.settings))
        if arguments == ("info", "--format", "{{json .}}"):
            return _json_line(_info(self.settings))
        if arguments[:2] == ("container", "ls"):
            return self._container_list(arguments)
        if arguments[:3] == ("container", "inspect", "--format"):
            return _json_line(self.containers[arguments[-1]])
        if arguments[:2] == ("volume", "ls"):
            return self._volume_list(arguments)
        if arguments[:3] == ("volume", "inspect", "--format"):
            return _json_line(self.volumes[arguments[-1]])
        raise AssertionError(f"unexpected Docker arguments: {arguments}")

    def _container_list(self, arguments):
        override = self._begin_lookup()
        name_filter, label_filters = _filters(arguments)
        name = (
            None
            if name_filter is None
            else name_filter.removeprefix("name=^/").removesuffix("$")
        )
        values = tuple(
            container_id
            for container_id, payload in sorted(self.containers.items())
            if (name is None or payload["Name"] == f"/{name}")
            and _labels_match(payload["Config"]["Labels"], label_filters)
        )
        return override if override is not None else _json_lines(values)

    def _volume_list(self, arguments):
        override = self._begin_lookup()
        name_filter, label_filters = _filters(arguments)
        name = (
            None
            if name_filter is None
            else name_filter.removeprefix("name=^").removesuffix("$")
        )
        values = tuple(
            volume_name
            for volume_name, payload in sorted(self.volumes.items())
            if (name is None or payload["Name"] == name)
            and _labels_match(payload["Labels"], label_filters)
        )
        return override if override is not None else _json_lines(values)

    def _begin_lookup(self):
        self.lookup_count += 1
        override = self.next_lookup_stdout
        self.next_lookup_stdout = None
        if self.lookup_count == self.mutate_after_lookup_count:
            self.lookup_mutation()
        return override


@pytest.fixture
def resource_context(tmp_path, monkeypatch):
    settings = CrossRunSettings.from_dict(
        load_config(_CANONICAL_CONFIG_PATH)["cross_run"]
    ).docker
    settings = replace(
        settings,
        runtime_executable_digest=tree_or_blob_digest(_TEST_DOCKER_BYTES),
    )

    def read_executable(_path, expected_digest):
        if expected_digest != settings.runtime_executable_digest:
            raise PinnedDockerRuntimeError(
                "Docker authority executable differs from its pinned digest"
            )
        return _TEST_DOCKER_BYTES

    monkeypatch.setattr(
        runtime_module,
        "read_verified_root_executable",
        read_executable,
    )
    monkeypatch.setattr(runtime_module, "_require_runtime_socket", lambda _path: None)
    tmp_path.chmod(0o700)
    runner = _InventoryDockerRunner(settings)
    runtime = PinnedDockerRuntime(
        trusted_root=tmp_path.resolve(),
        settings=settings,
        process_runner=runner,
    )
    claim = _claim()
    allocation = RunActionPreparationAllocation.mint(
        preparation_claim=claim,
        runtime_volume_authority=issue_runtime_volume_authority(claim, "a" * 32),
    )
    return DockerRunActionResourceManager(runtime), runner, allocation, claim


def _json_line(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode() + b"\n"


def _json_lines(values):
    return b"".join(_json_line(value) for value in values)


def _filters(arguments):
    filters = tuple(
        arguments[position + 1]
        for position, argument in enumerate(arguments)
        if argument == "--filter"
    )
    names = tuple(value for value in filters if value.startswith("name="))
    labels = tuple(
        value.removeprefix("label=") for value in filters if value.startswith("label=")
    )
    if len(names) > 1:
        raise AssertionError("at most one name filter is allowed")
    return (None if not names else names[0]), labels


def _labels_match(observed, filters):
    return all(
        observed.get(key) == value
        for key, value in (item.split("=", 1) for item in filters)
    )


def _label_mapping(labels):
    return {label.key: label.value for label in labels}


def _install_exact_resources(runner, allocation):
    claim = allocation.preparation_claim
    authority = allocation.runtime_volume_authority
    keeper_name = preparation_keeper_container_name(claim)
    main_name = preparation_container_name(claim)
    volume_name = preparation_volume_name(claim)
    runner.containers[_KEEPER_CONTAINER_ID] = {
        "Config": {
            "Labels": _label_mapping(preparation_keeper_container_labels(claim))
        },
        "Id": _KEEPER_CONTAINER_ID,
        "Name": f"/{keeper_name}",
    }
    runner.containers[_MAIN_CONTAINER_ID] = {
        "Config": {"Labels": _label_mapping(preparation_container_labels(claim))},
        "Id": _MAIN_CONTAINER_ID,
        "Name": f"/{main_name}",
    }
    runner.volumes[volume_name] = {
        "CreatedAt": "2026-07-25T00:00:00Z",
        "Labels": _label_mapping(
            preparation_volume_labels(claim, authority.generation_nonce)
        ),
        "Name": volume_name,
    }


def test_inventory_proves_all_three_names_absent_across_two_scans(resource_context):
    manager, runner, allocation, _claim = resource_context

    inventory = manager.observe(allocation)

    assert inventory.is_absent
    assert inventory.preparation_allocation == allocation
    assert runner.lookup_count == 18


@pytest.mark.parametrize(
    "arguments",
    (
        ("container", "stop", _MAIN_CONTAINER_ID),
        ("container", "kill", _MAIN_CONTAINER_ID),
        ("volume", "rm", "provider-volume"),
    ),
)
def test_resource_manager_retains_only_read_only_docker_authority(
    resource_context,
    arguments,
):
    manager, runner, _allocation, _claim = resource_context
    request_count = len(runner.requests)
    observation_authority = resources_module._run_action_observation_authority(manager)

    assert manager.runtime_settings == runner.settings
    assert not hasattr(manager, "runtime")
    assert not hasattr(manager, "_runtime")
    assert not hasattr(manager, "_observation_authority")
    assert not hasattr(observation_authority, "run_bounded")
    with pytest.raises(
        PinnedDockerRuntimeError,
        match="cannot execute provider mutations",
    ):
        observation_authority.run_control(arguments)
    assert len(runner.requests) == request_count


def test_resource_manager_runtime_binding_cannot_be_relabelled(resource_context):
    manager, runner, allocation, _claim = resource_context
    foreign_settings = replace(
        runner.settings,
        runtime_socket_path="/run/foreign-docker.sock",
    )
    manager._runtime_settings = foreign_settings
    manager._observation_authority = object()

    assert manager.runtime_settings == runner.settings
    assert manager.observe(allocation).is_absent


def test_inventory_rebinds_exact_labels_and_container_ids(resource_context):
    manager, runner, allocation, claim = resource_context
    _install_exact_resources(runner, allocation)

    inventory = manager.observe(allocation)

    assert inventory.volume_present is True
    assert inventory.keeper_container_id == _KEEPER_CONTAINER_ID
    assert inventory.main_container_id == _MAIN_CONTAINER_ID
    assert manager.inspect_volume(inventory)["Name"] == preparation_volume_name(claim)
    assert manager.inspect_keeper(inventory)["Id"] == _KEEPER_CONTAINER_ID
    assert manager.inspect_main(inventory)["Id"] == _MAIN_CONTAINER_ID
    inspected_targets = tuple(
        request.argv[-1]
        for request in runner.requests
        if request.argv[-5:-3] == ("container", "inspect")
    )
    assert _KEEPER_CONTAINER_ID in inspected_targets
    assert _MAIN_CONTAINER_ID in inspected_targets
    assert preparation_keeper_container_name(claim) not in inspected_targets
    assert preparation_container_name(claim) not in inspected_targets


@pytest.mark.parametrize("resource_kind", ("volume", "keeper", "main"))
def test_name_owned_by_wrong_labels_is_substitution(resource_context, resource_kind):
    manager, runner, allocation, claim = resource_context
    _install_exact_resources(runner, allocation)
    if resource_kind == "volume":
        runner.volumes[preparation_volume_name(claim)]["Labels"] = {
            "untrusted": "volume"
        }
    else:
        container_id = (
            _KEEPER_CONTAINER_ID if resource_kind == "keeper" else _MAIN_CONTAINER_ID
        )
        runner.containers[container_id]["Config"]["Labels"] = {
            "untrusted": resource_kind
        }

    with pytest.raises(DockerRunActionResourceError, match="labels are conflicted"):
        manager.observe(allocation)


def test_volume_from_another_generation_of_same_claim_is_substitution(
    resource_context,
):
    manager, runner, allocation, claim = resource_context
    _install_exact_resources(runner, allocation)
    other_allocation = RunActionPreparationAllocation.mint(
        preparation_claim=claim,
        runtime_volume_authority=issue_runtime_volume_authority(claim, "b" * 32),
    )
    runner.volumes[preparation_volume_name(claim)]["Labels"] = _label_mapping(
        preparation_volume_labels(
            claim,
            other_allocation.runtime_volume_authority.generation_nonce,
        )
    )

    with pytest.raises(DockerRunActionResourceError, match="labels are conflicted"):
        manager.observe(allocation)


def test_extra_label_is_rejected_after_label_filter_match(resource_context):
    manager, runner, allocation, _claim = resource_context
    _install_exact_resources(runner, allocation)
    runner.containers[_MAIN_CONTAINER_ID]["Config"]["Labels"]["extra"] = "substitution"

    with pytest.raises(
        DockerRunActionResourceError,
        match="differs from exact name and labels",
    ):
        manager.observe(allocation)


@pytest.mark.parametrize("resource_kind", ("volume", "keeper", "main"))
def test_exact_labels_under_another_name_are_conflicted(
    resource_context,
    resource_kind,
):
    manager, runner, allocation, claim = resource_context
    if resource_kind == "volume":
        authority = allocation.runtime_volume_authority
        runner.volumes["another-volume"] = {
            "CreatedAt": "2026-07-25T00:00:00Z",
            "Labels": _label_mapping(
                preparation_volume_labels(claim, authority.generation_nonce)
            ),
            "Name": "another-volume",
        }
    else:
        container_id = (
            _KEEPER_CONTAINER_ID if resource_kind == "keeper" else _MAIN_CONTAINER_ID
        )
        labels = (
            preparation_keeper_container_labels(claim)
            if resource_kind == "keeper"
            else preparation_container_labels(claim)
        )
        runner.containers[container_id] = {
            "Config": {"Labels": _label_mapping(labels)},
            "Id": container_id,
            "Name": f"/another-{resource_kind}",
        }

    with pytest.raises(DockerRunActionResourceError, match="labels are conflicted"):
        manager.observe(allocation)


def test_duplicate_role_labels_are_conflicted(resource_context):
    manager, runner, allocation, claim = resource_context
    _install_exact_resources(runner, allocation)
    runner.containers["c" * 64] = {
        "Config": {
            "Labels": _label_mapping(preparation_keeper_container_labels(claim))
        },
        "Id": "c" * 64,
        "Name": "/duplicate-keeper",
    }

    with pytest.raises(DockerRunActionResourceError, match="labels are conflicted"):
        manager.observe(allocation)


@pytest.mark.parametrize(
    "payload",
    (
        b'"' + b"a" * 64 + b'"',
        b'"' + b"a" * 64 + b'"\r\n',
        b'"' + b"a" * 64 + b'"\n"' + b"a" * 64 + b'"\n',
        b'"short"\n',
        b"42\n",
    ),
)
def test_malformed_or_ambiguous_lookup_fails_loud(resource_context, payload):
    manager, runner, allocation, _claim = resource_context
    runner.next_lookup_stdout = payload

    with pytest.raises(DockerRunActionResourceError):
        manager.observe(allocation)


def test_resource_change_between_complete_scans_is_rejected(resource_context):
    manager, runner, allocation, claim = resource_context
    authority = allocation.runtime_volume_authority
    volume_name = preparation_volume_name(claim)

    def add_volume():
        runner.volumes[volume_name] = {
            "CreatedAt": "2026-07-25T00:00:00Z",
            "Labels": _label_mapping(
                preparation_volume_labels(claim, authority.generation_nonce)
            ),
            "Name": volume_name,
        }

    runner.mutate_after_lookup_count = 9
    runner.lookup_mutation = add_volume

    with pytest.raises(DockerRunActionResourceError, match="changed during inventory"):
        manager.observe(allocation)


def test_volume_recreation_between_scans_changes_occurrence_digest(resource_context):
    manager, runner, allocation, claim = resource_context
    _install_exact_resources(runner, allocation)
    volume_name = preparation_volume_name(claim)

    def recreate_volume():
        runner.volumes[volume_name]["CreatedAt"] = "2026-07-25T00:00:01Z"

    runner.mutate_after_lookup_count = 9
    runner.lookup_mutation = recreate_volume

    with pytest.raises(DockerRunActionResourceError, match="changed during inventory"):
        manager.observe(allocation)


def test_volume_change_after_rebinding_is_rejected(resource_context):
    manager, runner, allocation, claim = resource_context
    _install_exact_resources(runner, allocation)
    inventory = manager.observe(allocation)
    volume_name = preparation_volume_name(claim)

    def recreate_volume():
        runner.volumes[volume_name]["CreatedAt"] = "2026-07-25T00:00:01Z"

    runner.mutate_after_lookup_count = runner.lookup_count + 18
    runner.lookup_mutation = recreate_volume

    with pytest.raises(DockerRunActionResourceError, match="changed after inventory"):
        manager.inspect_volume(inventory)


def test_stale_inventory_cannot_be_reused_after_resource_change(resource_context):
    manager, runner, allocation, _claim = resource_context
    _install_exact_resources(runner, allocation)
    inventory = manager.observe(allocation)
    del runner.containers[_MAIN_CONTAINER_ID]

    with pytest.raises(
        DockerRunActionResourceError,
        match="changed before inspection",
    ):
        manager.inspect_keeper(inventory)
