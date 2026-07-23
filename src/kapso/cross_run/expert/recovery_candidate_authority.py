"""Process-local authority for clean-forward recovery candidate admission."""

from __future__ import annotations

import os


class ExpertRecoveryCandidateAuthorityError(ValueError):
    """A recovery candidate capability is foreign or malformed."""


_RECOVERY_CANDIDATE_AUTHORITY_SEAL = object()


class ExpertRecoveryCandidateAuthority:
    """Store-bound authority wrapper owned by one recovery coordinator."""

    __slots__ = (
        "_candidate_store",
        "_coordinator",
        "_owner_process_id",
        "_proposal_engine",
    )

    def __init__(
        self,
        seal: object,
        *,
        coordinator: object,
        candidate_store: object,
        proposal_engine: object,
    ) -> None:
        if seal is not _RECOVERY_CANDIDATE_AUTHORITY_SEAL:
            raise ExpertRecoveryCandidateAuthorityError(
                "recovery candidate authority is not coordinator sealed"
            )
        object.__setattr__(self, "_coordinator", coordinator)
        object.__setattr__(self, "_candidate_store", candidate_store)
        object.__setattr__(self, "_proposal_engine", proposal_engine)
        object.__setattr__(self, "_owner_process_id", os.getpid())

    def __setattr__(self, name: str, value: object) -> None:
        raise ExpertRecoveryCandidateAuthorityError(
            "recovery candidate authority is immutable"
        )

    def __reduce__(self) -> object:
        raise ExpertRecoveryCandidateAuthorityError(
            "recovery candidate authority cannot be serialized"
        )

    def __reduce_ex__(self, protocol: int) -> object:
        raise ExpertRecoveryCandidateAuthorityError(
            "recovery candidate authority cannot be serialized"
        )

    def _require_bound(self, *, candidate_store: object) -> None:
        if (
            self._owner_process_id != os.getpid()
            or self._candidate_store is not candidate_store
        ):
            raise ExpertRecoveryCandidateAuthorityError(
                "recovery candidate authority is foreign"
            )

    def _require_proposal_engine(self, *, proposal_engine: object) -> None:
        if (
            self._owner_process_id != os.getpid()
            or self._proposal_engine is not proposal_engine
        ):
            raise ExpertRecoveryCandidateAuthorityError(
                "recovery proposal authority is foreign"
            )

    def _finalize_under_store_lock(
        self,
        *,
        candidate_store: object,
        selection: object,
        closure: object,
        commit_record: object,
        barrier_replay_basis: object,
    ) -> object:
        self._require_bound(candidate_store=candidate_store)
        return self._coordinator._finalize_recovery_admission_under_store_lock(
            selection=selection,
            closure=closure,
            commit_record=commit_record,
            barrier_replay_basis=barrier_replay_basis,
        )


def _seal_expert_recovery_candidate_authority(
    *,
    coordinator: object,
    candidate_store: object,
    proposal_engine: object,
) -> ExpertRecoveryCandidateAuthority:
    return ExpertRecoveryCandidateAuthority(
        _RECOVERY_CANDIDATE_AUTHORITY_SEAL,
        coordinator=coordinator,
        candidate_store=candidate_store,
        proposal_engine=proposal_engine,
    )


__all__ = [
    "ExpertRecoveryCandidateAuthority",
    "ExpertRecoveryCandidateAuthorityError",
]
