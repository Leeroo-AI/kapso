from __future__ import annotations

import copy
import os
import pickle
import select
import signal
from dataclasses import FrozenInstanceError

import pytest

import kapso.cross_run.launch.run_action_docker_adapter as adapter_module
from kapso.core.config import load_config
from kapso.cross_run.docker.runtime import PinnedDockerRuntime
from kapso.cross_run.launch.run_action_contracts import (
    RunFrontierActionKind,
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_docker_adapter import (
    DockerRunActionAdapterError,
    DockerRunActionExecutionAdapter,
)
from kapso.cross_run.launch.run_action_docker_projection import (
    DockerRunActionCommand,
)
from kapso.cross_run.launch.run_action_store import RunActionExecutionEvent
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionPreparationAllocation,
)
from kapso.cross_run.settings import CrossRunSettings
from test_run_action_docker_projection import _policy
from test_run_action_supervisor_contracts import (
    _claim,
    _remint_contract,
    _volume_authority,
)
from test_run_frontier_action_gate import _boundary_identity

_CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"
_COMMAND = DockerRunActionCommand.build(
    entrypoint="/bin/tool",
    arguments=("offline",),
)
_PREDECESSOR_EVENT_ID = (
    f"{RunActionExecutionEvent.CONTENT_NAMESPACE}:sha256:" + "0" * 64
)


class _NarrowManager:
    def __init__(self, runtime=None, **arguments) -> None:
        selected_runtime = runtime
        if selected_runtime is None:
            selected_runtime = arguments["runtime"]
        self.runtime_settings = selected_runtime.settings


def _adapter_case(monkeypatch):
    settings = CrossRunSettings.from_dict(
        load_config(_CANONICAL_CONFIG_PATH)["cross_run"]
    )
    policy = _policy(
        settings.docker,
        workspace_access=RunFrontierWorkspaceAccess.NONE,
        command_template_id=_COMMAND.command_template_id,
    )
    base_boundary = _boundary_identity(RunFrontierActionKind.EMBEDDING)
    lifecycle_identity = _remint_contract(
        base_boundary.execution_lifecycle_identity,
        execution_policy_id=policy.docker_execution_policy_id,
    )
    runtime = object.__new__(PinnedDockerRuntime)
    runtime._settings = settings.docker
    for manager_name in (
        "DockerRunActionResourceManager",
        "DockerRunActionPreparationManager",
        "DockerRunActionStartManager",
        "DockerRunActionContainmentManager",
        "DockerRunActionCredentialRetirementManager",
    ):
        monkeypatch.setattr(adapter_module, manager_name, _NarrowManager)
    monkeypatch.setattr(
        adapter_module,
        "observe_supervisor_helper",
        lambda _policy: object(),
    )
    monkeypatch.setattr(
        adapter_module,
        "observe_docker_init_source",
        lambda _policy: object(),
    )
    adapter = DockerRunActionExecutionAdapter(
        execution_lifecycle_identity=lifecycle_identity,
        execution_policy=policy,
        command=_COMMAND,
        runtime=runtime,
        launch_settings=settings.launch,
    )
    claim = _claim(policy=policy)
    allocation = RunActionPreparationAllocation.mint(
        preparation_claim=claim,
        runtime_volume_authority=_volume_authority(claim, nonce="8" * 32),
    )
    return adapter, allocation, lifecycle_identity, policy


def test_adapter_seals_private_collaborators_and_rejects_copy(
    monkeypatch,
) -> None:
    adapter, allocation, _identity, policy = _adapter_case(monkeypatch)

    assert not hasattr(adapter, "_command")
    assert not hasattr(adapter, "_resource_manager")
    assert (
        adapter.prepared_event_size_bound(
            preparation_allocation=allocation,
            predecessor_event_id=_PREDECESSOR_EVENT_ID,
        )
        > 0
    )
    with pytest.raises(FrozenInstanceError):
        adapter.execution_policy = policy
    with pytest.raises(DockerRunActionAdapterError, match="cannot be copied"):
        copy.copy(adapter)
    with pytest.raises(DockerRunActionAdapterError, match="cannot be copied"):
        copy.deepcopy(adapter)
    with pytest.raises(DockerRunActionAdapterError, match="cannot be serialized"):
        pickle.dumps(adapter)


def test_adapter_rejects_public_policy_shadowing_and_forked_authority(
    monkeypatch,
) -> None:
    adapter, allocation, _identity, policy = _adapter_case(monkeypatch)
    object.__setattr__(
        adapter,
        "execution_policy",
        _remint_contract(policy, hostname="substituted"),
    )
    with pytest.raises(DockerRunActionAdapterError, match="substituted or foreign"):
        adapter.prepared_event_size_bound(
            preparation_allocation=allocation,
            predecessor_event_id=_PREDECESSOR_EVENT_ID,
        )

    adapter, allocation, _identity, _policy = _adapter_case(monkeypatch)
    owner_process_id = adapter_module.os.getpid()
    monkeypatch.setattr(adapter_module.os, "getpid", lambda: owner_process_id + 1)
    with pytest.raises(DockerRunActionAdapterError, match="substituted or foreign"):
        adapter.prepared_event_size_bound(
            preparation_allocation=allocation,
            predecessor_event_id=_PREDECESSOR_EVENT_ID,
        )


def test_adapter_rejects_fork_before_inherited_state_lock(
    monkeypatch,
) -> None:
    adapter, allocation, _identity, _policy = _adapter_case(monkeypatch)
    settings = CrossRunSettings.from_dict(
        load_config(_CANONICAL_CONFIG_PATH)["cross_run"]
    )
    read_descriptor, write_descriptor = os.pipe()
    with adapter_module._ADAPTER_STATE_LOCK:
        child_process_id = os.fork()
        if child_process_id == 0:
            os.close(read_descriptor)
            with pytest.raises(
                DockerRunActionAdapterError,
                match="substituted or foreign",
            ):
                adapter.prepared_event_size_bound(
                    preparation_allocation=allocation,
                    predecessor_event_id=_PREDECESSOR_EVENT_ID,
                )
            os.write(write_descriptor, b"rejected-before-lock")
            os.close(write_descriptor)
            os._exit(0)
        os.close(write_descriptor)
        readable, _writable, _exceptional = select.select(
            (read_descriptor,),
            (),
            (),
            settings.docker.command_timeout_seconds,
        )
        if not readable:
            os.kill(child_process_id, signal.SIGKILL)
        payload = os.read(read_descriptor, 64) if readable else b""
    os.close(read_descriptor)
    _waited_process_id, wait_status = os.waitpid(child_process_id, 0)

    assert readable == [read_descriptor]
    assert payload == b"rejected-before-lock"
    assert os.WIFEXITED(wait_status)
    assert os.WEXITSTATUS(wait_status) == 0


@pytest.mark.parametrize(
    ("status", "expected"),
    (
        ("created", adapter_module._DockerMainLifecycle.INERT),
        ("running", adapter_module._DockerMainLifecycle.RUNNING),
        ("exited", adapter_module._DockerMainLifecycle.EXITED),
    ),
)
def test_main_lifecycle_dispatch_is_closed(status, expected) -> None:
    assert adapter_module._main_lifecycle({"State": {"Status": status}}) is expected


@pytest.mark.parametrize(
    "raw_main",
    (
        {},
        {"State": None},
        {"State": {}},
        {"State": {"Status": 1}},
        {"State": {"Status": "paused"}},
    ),
)
def test_main_lifecycle_rejects_malformed_or_unadmitted_state(raw_main) -> None:
    with pytest.raises(DockerRunActionAdapterError):
        adapter_module._main_lifecycle(raw_main)
