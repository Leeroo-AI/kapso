"""Durable spawn-commit contract shared by the action store and supervisor."""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from typing import ClassVar

from kapso.cross_run.canonical import require_content_id, require_identifier
from kapso.cross_run.contracts import StrictContract
from kapso.cross_run.launch.run_action_contracts import (
    RunActionBoundaryIdentity,
    RunActionContractError,
)
from kapso.cross_run.launch.run_action_reservation_contracts import (
    RunActionReservation,
)

_NONCE_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class RunActionSpawnContractError(RunActionContractError):
    """A durable spawn commit is malformed or internally inconsistent."""


@dataclass(frozen=True)
class RunActionSpawnCommit(StrictContract):
    """Pre-spawn durable fence for one exact provider invocation."""

    spawn_commit_id: str
    reservation_id: str
    provider_execution_id: str
    invocation_nonce: str
    security_observation_id: str
    boundary_identity: RunActionBoundaryIdentity

    CONTENT_NAMESPACE: ClassVar[str] = "run-action-spawn-commit"
    IDENTITY_FIELD: ClassVar[str] = "spawn_commit_id"

    def _validate(self) -> None:
        _require_namespaced_id(
            self.reservation_id,
            RunActionReservation.CONTENT_NAMESPACE,
            "run action spawn reservation",
        )
        require_identifier(
            self.provider_execution_id,
            "run action provider execution ID",
        )
        _require_namespaced_id(
            self.security_observation_id,
            "security-denylist-observation",
            "run action spawn security observation",
        )
        if _NONCE_PATTERN.fullmatch(self.invocation_nonce) is None:
            raise RunActionSpawnContractError(
                "run action spawn nonce must be 128-bit lowercase hex"
            )
        if type(self.boundary_identity) is not RunActionBoundaryIdentity:
            raise RunActionSpawnContractError(
                "run action spawn requires one exact boundary identity"
            )

    @classmethod
    def build(
        cls,
        *,
        reservation_id: str,
        provider_execution_id: str,
        security_observation_id: str,
        boundary_identity: RunActionBoundaryIdentity,
    ) -> "RunActionSpawnCommit":
        return cls.mint(
            reservation_id=reservation_id,
            provider_execution_id=provider_execution_id,
            invocation_nonce=secrets.token_hex(16),
            security_observation_id=security_observation_id,
            boundary_identity=boundary_identity,
        )


def _require_namespaced_id(value: str, namespace: str, name: str) -> None:
    require_content_id(value, name)
    if value.split(":sha256:", 1)[0] != namespace:
        raise RunActionSpawnContractError(f"{name} uses another namespace")


__all__ = [
    "RunActionSpawnCommit",
    "RunActionSpawnContractError",
]
