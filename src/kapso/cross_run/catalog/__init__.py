"""Immutable cross-run catalog projection, interpretation, and admission."""

from kapso.cross_run.catalog.admission import (
    AdmissionReducer,
    CatalogRevocation,
    CatalogTaint,
    ClaimEvidenceClosure,
)
from kapso.cross_run.catalog.agent_operations import CatalogAgentOperationRecord
from kapso.cross_run.catalog.assertions import ReviewRegistry
from kapso.cross_run.catalog.claims import ClaimProposer, ClaimProposalPacket
from kapso.cross_run.catalog.projector import (
    BundleProjectionManifest,
    ProjectionResult,
    RunBundleProjector,
)
from kapso.cross_run.catalog.reducer import CatalogGenerationReducer
from kapso.cross_run.catalog.reviews import (
    CatalogReviewer,
    CatalogReviewPacket,
)
from kapso.cross_run.catalog.service import CrossRunCatalog
from kapso.cross_run.catalog.store import CatalogStore

__all__ = [
    "AdmissionReducer",
    "BundleProjectionManifest",
    "CatalogAgentOperationRecord",
    "CatalogGenerationReducer",
    "CatalogRevocation",
    "CatalogReviewer",
    "CatalogReviewPacket",
    "CatalogStore",
    "CatalogTaint",
    "ClaimEvidenceClosure",
    "ClaimProposer",
    "ClaimProposalPacket",
    "CrossRunCatalog",
    "ProjectionResult",
    "ReviewRegistry",
    "RunBundleProjector",
]
