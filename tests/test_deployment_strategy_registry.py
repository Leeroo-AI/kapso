"""The public deploy surface cannot bypass the pinned Docker authority."""

import importlib.util

from kapso.deployment import DeployStrategy, DeploymentFactory


def test_shared_socket_docker_strategy_is_not_discoverable():
    assert "docker" not in DeploymentFactory.list_strategies()
    assert "DOCKER" not in DeployStrategy.__members__
    assert importlib.util.find_spec("kapso.deployment.strategies.docker.runner") is None
