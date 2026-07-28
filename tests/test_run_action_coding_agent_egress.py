from dataclasses import replace
import socket
from pathlib import Path

import pytest

from kapso.core.config import load_config
from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.launch.run_action_coding_agent_egress import (
    NativeCodingAgentEgressBroker,
    NativeCodingAgentEgressError,
)
from kapso.cross_run.settings import CrossRunSettings

_CONFIG_PATH = "src/kapso/config.yaml"


def _settings():
    settings = CrossRunSettings.from_dict(load_config(_CONFIG_PATH)["cross_run"])
    return replace(
        settings.launch, coding_agent_egress_broker_socket_path="broker.sock"
    )


def _short_state_root(tmp_path: Path) -> Path:
    digest = tree_or_blob_digest(tmp_path.as_posix().encode()).rsplit(":", 1)[1]
    return (Path("/tmp") / f"kapso-egress-{digest[:16]}").resolve()


def test_native_egress_broker_starts_from_and_returns_to_exact_absence(tmp_path: Path):
    settings = _settings()
    state_root = _short_state_root(tmp_path)
    broker = NativeCodingAgentEgressBroker(
        settings=settings,
        state_root=state_root,
    )
    socket_path = broker.socket_path

    assert socket_path.is_socket()
    broker.close()
    assert not socket_path.exists()


def test_native_egress_broker_rejects_an_existing_live_socket(tmp_path: Path):
    settings = _settings()
    state_root = _short_state_root(tmp_path)
    socket_path = state_root / settings.coding_agent_egress_broker_socket_path
    socket_path.parent.mkdir(parents=True, mode=0o700)
    socket_path.parent.chmod(0o700)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(socket_path.as_posix())
    socket_path.chmod(0o600)
    listener.listen(1)

    with pytest.raises(
        NativeCodingAgentEgressError,
        match="already active",
    ):
        NativeCodingAgentEgressBroker(
            settings=settings,
            state_root=state_root,
        )

    listener.close()
    socket_path.unlink()
