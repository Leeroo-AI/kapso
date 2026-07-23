"""Process-local authority capability for expert release-use revocations."""

from __future__ import annotations

import os


class CatalogReleaseUseRevocationAuthorityError(ValueError):
    """A release-use author capability is foreign or malformed."""


_CATALOG_RELEASE_USE_REVOCATION_AUTHORITY_SEAL = object()


class CatalogReleaseUseRevocationAuthority:
    """Exact catalog-bound authority owned by one release-use author."""

    __slots__ = ("_author", "_catalog", "_owner_process_id")

    def __init__(
        self,
        seal: object,
        *,
        author: object,
        catalog: object,
    ) -> None:
        if seal is not _CATALOG_RELEASE_USE_REVOCATION_AUTHORITY_SEAL:
            raise CatalogReleaseUseRevocationAuthorityError(
                "release-use revocation authority is not author sealed"
            )
        object.__setattr__(self, "_author", author)
        object.__setattr__(self, "_catalog", catalog)
        object.__setattr__(self, "_owner_process_id", os.getpid())

    def __setattr__(self, name: str, value: object) -> None:
        raise CatalogReleaseUseRevocationAuthorityError(
            "release-use revocation authority is immutable"
        )

    def __reduce__(self) -> object:
        raise CatalogReleaseUseRevocationAuthorityError(
            "release-use revocation authority cannot be serialized"
        )

    def __reduce_ex__(self, protocol: int) -> object:
        raise CatalogReleaseUseRevocationAuthorityError(
            "release-use revocation authority cannot be serialized"
        )

    def _require_bound(self, *, catalog: object) -> None:
        if self._owner_process_id != os.getpid() or self._catalog is not catalog:
            raise CatalogReleaseUseRevocationAuthorityError(
                "release-use revocation authority is foreign"
            )

    def _require_authenticated_event(
        self,
        *,
        catalog: object,
        historical_activation: object,
        event: object,
    ) -> None:
        self._require_bound(catalog=catalog)
        self._author._require_authenticated_event(
            historical_activation=historical_activation,
            event=event,
        )


def _seal_catalog_release_use_revocation_authority(
    *,
    author: object,
    catalog: object,
) -> CatalogReleaseUseRevocationAuthority:
    return CatalogReleaseUseRevocationAuthority(
        _CATALOG_RELEASE_USE_REVOCATION_AUTHORITY_SEAL,
        author=author,
        catalog=catalog,
    )


__all__ = [
    "CatalogReleaseUseRevocationAuthority",
    "CatalogReleaseUseRevocationAuthorityError",
]
