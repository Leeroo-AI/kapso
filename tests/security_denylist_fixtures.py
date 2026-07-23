from __future__ import annotations

from kapso.cross_run.canonical import content_id
from kapso.cross_run.contracts import (
    SecurityDenylistKind,
    SecurityDenylistRevocation,
)


def matched_security_revocations(
    subject_ids: tuple[str, ...],
) -> tuple[SecurityDenylistRevocation, ...]:
    return tuple(
        sorted(
            (
                SecurityDenylistRevocation.mint(
                    subject_id=subject_id,
                    kind=SecurityDenylistKind.SECURITY,
                    reason_code="verified_compromise",
                    evidence_ids=(
                        content_id(
                            "security-denylist-evidence",
                            {"subject_id": subject_id},
                        ),
                    ),
                    recorded_at="2026-07-21T12:00:00Z",
                )
                for subject_id in subject_ids
            ),
            key=lambda revocation: revocation.revocation_id,
        )
    )
