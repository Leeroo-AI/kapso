"""Single typed registry for records crossing catalog and knowledge boundaries."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from types import MappingProxyType
from typing import Any, Mapping

from kapso.cross_run.canonical import canonical_json_bytes, require_identifier
from kapso.cross_run.catalog.store import (
    CatalogGenerationManifest,
    CatalogInputDelta,
)
from kapso.cross_run.contracts import (
    CatalogEntryState,
    CodingAgentOperationReceipt,
    ContractValidationError,
    ExpertScopeContract,
    KnowledgeClaim,
    PriorIdea,
    ReviewAssertion,
    RunBundle,
    StrictContract,
    TransferEpisode,
)
from kapso.cross_run.record_contracts import (
    BundleProjectionManifest,
    CatalogAgentOperationRecord,
    CatalogRevocation,
    CatalogTaint,
    ClaimEvidenceClosure,
    ExpertReleaseUseRevocation,
    ExecutionRevisionEvent,
    SanitationReport,
)


def _typed_registry(
    record_types: tuple[type[StrictContract], ...],
) -> Mapping[str, type[StrictContract]]:
    registry: dict[str, type[StrictContract]] = {}
    for record_type in record_types:
        namespace = record_type.CONTENT_NAMESPACE
        identity_field = record_type.IDENTITY_FIELD
        if namespace is None or identity_field is None:
            raise ContractValidationError(
                "registered cross-run records must be content identified"
            )
        if namespace in registry:
            raise ContractValidationError(
                f"duplicate cross-run record namespace: {namespace}"
            )
        registry[namespace] = record_type
    return MappingProxyType(registry)


CATALOG_FACT_RECORD_TYPES = _typed_registry(
    (
        BundleProjectionManifest,
        CatalogRevocation,
        CatalogTaint,
        ClaimEvidenceClosure,
        ExpertReleaseUseRevocation,
        CatalogAgentOperationRecord,
        CodingAgentOperationReceipt,
        ExecutionRevisionEvent,
        KnowledgeClaim,
        PriorIdea,
        ReviewAssertion,
        RunBundle,
        SanitationReport,
        TransferEpisode,
    )
)

KNOWLEDGE_RECORD_TYPES = _typed_registry(
    (
        *CATALOG_FACT_RECORD_TYPES.values(),
        CatalogEntryState,
        CatalogGenerationManifest,
        CatalogInputDelta,
        ExpertScopeContract,
    )
)

KNOWLEDGE_RECORD_IDENTITY_FIELDS: Mapping[str, str] = MappingProxyType(
    {
        namespace: record_type.IDENTITY_FIELD
        for namespace, record_type in KNOWLEDGE_RECORD_TYPES.items()
    }
)


def parse_knowledge_record_payload(
    record_kind: str,
    payload: Mapping[str, Any],
) -> StrictContract:
    """Strictly parse one complete knowledge record and preserve canonical meaning."""

    require_identifier(record_kind, "record_kind")
    if not isinstance(payload, MappingABC):
        raise ContractValidationError("knowledge record payload must be an object")
    record_type = KNOWLEDGE_RECORD_TYPES.get(record_kind)
    if record_type is None:
        raise ContractValidationError("unknown knowledge record kind")
    record = record_type.from_dict(payload)
    if record.to_json_bytes() != canonical_json_bytes(payload):
        raise ContractValidationError(
            "knowledge record payload is not the exact canonical contract shape"
        )
    return record


def record_identity(record: StrictContract) -> str:
    """Return the declared identity of a registered content-addressed record."""

    identity_field = record.IDENTITY_FIELD
    if identity_field is None:
        raise ContractValidationError("cross-run record has no content identity")
    return getattr(record, identity_field)
