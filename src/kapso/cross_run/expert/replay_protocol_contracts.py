"""Source-replay allocation authority above the shared evaluator ABI."""

from __future__ import annotations

import re
from dataclasses import dataclass

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    require_content_id,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import StrictContract


class ExpertSourceReplayProtocolError(ValueError):
    """A source-replay projection violates its exact execution protocol."""


@dataclass(frozen=True)
class ExpertSourceReplayInvocationAllocation(StrictContract):
    """Private journal allocation binding one unpredictable nonce to one leg."""

    reservation_id: str
    execution_case_id: str
    execution_leg_id: str
    invocation_nonce: str

    def _validate(self) -> None:
        for value, namespace, name in (
            (
                self.reservation_id,
                "expert-source-replay-execution-reservation",
                "reservation_id",
            ),
            (
                self.execution_case_id,
                "expert-source-replay-execution-case",
                "execution_case_id",
            ),
            (
                self.execution_leg_id,
                "expert-source-replay-execution-leg",
                "execution_leg_id",
            ),
        ):
            require_content_id(value, f"source replay allocation {name}")
            if value.split(":sha256:", 1)[0] != namespace:
                raise ExpertSourceReplayProtocolError(
                    f"source replay allocation {name} uses the wrong namespace"
                )
        if (
            not isinstance(self.invocation_nonce, str)
            or re.fullmatch(r"[0-9a-f]{32}", self.invocation_nonce) is None
        ):
            raise ExpertSourceReplayProtocolError(
                "source replay allocation nonce must contain 128 random bits"
            )

    @property
    def opaque_invocation_id(self) -> str:
        digest = tree_or_blob_digest(
            canonical_json_bytes(
                {
                    "execution_case_id": self.execution_case_id,
                    "execution_leg_id": self.execution_leg_id,
                    "invocation_nonce": self.invocation_nonce,
                    "reservation_id": self.reservation_id,
                }
            )
        ).removeprefix("sha256:")
        return f"task_evaluation_invocation_{digest[:32]}"
