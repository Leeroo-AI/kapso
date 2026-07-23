"""Process-local authority capabilities for composition admission."""

from __future__ import annotations

import os


class ExpertCompositionAdmissionAuthorityError(ValueError):
    """A composition admission capability is foreign, inactive, or malformed."""


_COMPOSITION_ADMISSION_AUTHORITY_SEAL = object()
_COMPOSITION_APPROVAL_LEASE_SEAL = object()


class ExpertCompositionAdmissionAuthority:
    """Exact store-bound authority wrapper owned by one coordinator instance."""

    __slots__ = ("_candidate_store", "_coordinator", "_owner_process_id")

    def __init__(
        self,
        seal: object,
        *,
        coordinator: object,
        candidate_store: object,
    ) -> None:
        if seal is not _COMPOSITION_ADMISSION_AUTHORITY_SEAL:
            raise ExpertCompositionAdmissionAuthorityError(
                "composition admission authority is not coordinator sealed"
            )
        object.__setattr__(self, "_coordinator", coordinator)
        object.__setattr__(self, "_candidate_store", candidate_store)
        object.__setattr__(self, "_owner_process_id", os.getpid())

    def __setattr__(self, name, value) -> None:
        raise ExpertCompositionAdmissionAuthorityError(
            "composition admission authority is immutable"
        )

    def __reduce__(self):
        raise ExpertCompositionAdmissionAuthorityError(
            "composition admission authority cannot be serialized"
        )

    def __reduce_ex__(self, protocol):
        raise ExpertCompositionAdmissionAuthorityError(
            "composition admission authority cannot be serialized"
        )

    def _require_bound(self, *, candidate_store: object) -> None:
        if (
            self._owner_process_id != os.getpid()
            or self._candidate_store is not candidate_store
        ):
            raise ExpertCompositionAdmissionAuthorityError(
                "composition admission authority is foreign"
            )

    def _require_approval_lease(
        self,
        *,
        candidate_store: object,
        approval_lease: ExpertCompositionApprovalLease,
        approved_sources: tuple[object, ...],
    ) -> None:
        self._require_bound(candidate_store=candidate_store)
        if type(approval_lease) is not ExpertCompositionApprovalLease:
            raise ExpertCompositionAdmissionAuthorityError(
                "composition admission lacks its exact approval lease"
            )
        approval_lease._require_active(
            resolver=self._coordinator.source_resolver,
            approved_sources=approved_sources,
        )

    def _finalize_under_store_lock(
        self,
        *,
        candidate_store: object,
        freshness_context: object,
        closure: object,
        commit_record: object,
    ) -> object:
        self._require_bound(candidate_store=candidate_store)
        return self._coordinator._finalize_composition_admission_under_store_lock(
            freshness_context=freshness_context,
            closure=closure,
            commit_record=commit_record,
        )


class ExpertCompositionApprovalLease:
    """Active shared validation lease over one exact approved source tuple."""

    __slots__ = (
        "_active",
        "_approved_sources",
        "_owner_process_id",
        "_resolver",
    )

    def __init__(
        self,
        seal: object,
        *,
        resolver: object,
        approved_sources: tuple[object, ...],
    ) -> None:
        if seal is not _COMPOSITION_APPROVAL_LEASE_SEAL:
            raise ExpertCompositionAdmissionAuthorityError(
                "composition approval lease is not resolver sealed"
            )
        object.__setattr__(self, "_resolver", resolver)
        object.__setattr__(self, "_approved_sources", approved_sources)
        object.__setattr__(self, "_owner_process_id", os.getpid())
        object.__setattr__(self, "_active", True)

    def __setattr__(self, name, value) -> None:
        raise ExpertCompositionAdmissionAuthorityError(
            "composition approval lease is immutable"
        )

    def __reduce__(self):
        raise ExpertCompositionAdmissionAuthorityError(
            "composition approval lease cannot be serialized"
        )

    def __reduce_ex__(self, protocol):
        raise ExpertCompositionAdmissionAuthorityError(
            "composition approval lease cannot be serialized"
        )

    def _require_active(
        self,
        *,
        resolver: object,
        approved_sources: tuple[object, ...],
    ) -> None:
        same_sources = (
            type(approved_sources) is tuple
            and len(approved_sources) == len(self._approved_sources)
            and all(
                actual is sealed
                for actual, sealed in zip(
                    approved_sources,
                    self._approved_sources,
                )
            )
        )
        if (
            not self._active
            or self._owner_process_id != os.getpid()
            or self._resolver is not resolver
            or not same_sources
        ):
            raise ExpertCompositionAdmissionAuthorityError(
                "composition approval lease is inactive or foreign"
            )

    def _deactivate(self) -> None:
        object.__setattr__(self, "_active", False)


def _seal_expert_composition_admission_authority(
    *,
    coordinator: object,
    candidate_store: object,
) -> ExpertCompositionAdmissionAuthority:
    return ExpertCompositionAdmissionAuthority(
        _COMPOSITION_ADMISSION_AUTHORITY_SEAL,
        coordinator=coordinator,
        candidate_store=candidate_store,
    )


def _seal_expert_composition_approval_lease(
    *,
    resolver: object,
    approved_sources: tuple[object, ...],
) -> ExpertCompositionApprovalLease:
    return ExpertCompositionApprovalLease(
        _COMPOSITION_APPROVAL_LEASE_SEAL,
        resolver=resolver,
        approved_sources=approved_sources,
    )
