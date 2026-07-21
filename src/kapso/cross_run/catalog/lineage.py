"""Exact immutable RunBundle lineage resolution and projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from kapso.cross_run.canonical import require_content_id
from kapso.cross_run.capture.bundle import RunBundleReader
from kapso.cross_run.capture.bundle_lineage import validate_run_bundle_root
from kapso.cross_run.catalog.projector import ProjectionResult, RunBundleProjector
from kapso.cross_run.contracts import RunBundle


class RunBundleLineageError(ValueError):
    """An exact bundle lineage is absent, cyclic, or exceeds policy."""


class ExactRunBundleSource(Protocol):
    """Resolve only a caller-named immutable bundle identity."""

    def read_exact(self, bundle_id: str) -> RunBundleReader | None: ...


@dataclass(frozen=True)
class VerifiedRunBundleLineage:
    """A root-to-tip bundle chain and its fully replayed tip projection."""

    bundle_ids: tuple[str, ...]
    tip_bundle: RunBundleReader
    tip_projection: ProjectionResult

    def __post_init__(self) -> None:
        if not self.bundle_ids:
            raise RunBundleLineageError("verified bundle lineage must not be empty")
        for bundle_id in self.bundle_ids:
            require_content_id(bundle_id, "lineage bundle_id")
        if len(self.bundle_ids) != len(set(self.bundle_ids)):
            raise RunBundleLineageError("verified bundle lineage contains a cycle")
        if (
            self.bundle_ids[-1] != self.tip_bundle.manifest.bundle_id
            or self.tip_projection.source_bundle != self.tip_bundle.manifest
        ):
            raise RunBundleLineageError(
                "verified projection does not belong to the lineage tip"
            )


class RunBundleLineageProvider:
    """Follow exact predecessor IDs, then project the proven root-to-tip chain."""

    def __init__(
        self,
        source: ExactRunBundleSource,
        projector: RunBundleProjector,
        maximum_bundles: int,
    ) -> None:
        if type(maximum_bundles) is not int or maximum_bundles <= 0:
            raise RunBundleLineageError("bundle lineage limit must be positive")
        self.source = source
        self.projector = projector
        self.maximum_bundles = maximum_bundles

    def resolve_exact(self, bundle_id: str) -> VerifiedRunBundleLineage:
        require_content_id(bundle_id, "bundle_id")
        reverse_bundle_ids: list[str] = []
        seen_ids: set[str] = set()
        current_id = bundle_id
        root_manifest: RunBundle | None = None
        while len(reverse_bundle_ids) < self.maximum_bundles:
            if current_id in seen_ids:
                raise RunBundleLineageError("bundle lineage contains a cycle")
            seen_ids.add(current_id)
            bundle = self.source.read_exact(current_id)
            if bundle is None:
                raise RunBundleLineageError(
                    f"bundle lineage is missing exact predecessor: {current_id}"
                )
            if bundle.manifest.bundle_id != current_id:
                raise RunBundleLineageError(
                    "exact bundle source returned another bundle identity"
                )
            reverse_bundle_ids.append(current_id)
            predecessor_id = bundle.manifest.supersedes_bundle_id
            if predecessor_id is None:
                root_manifest = bundle.manifest
                break
            require_content_id(predecessor_id, "supersedes_bundle_id")
            if predecessor_id in seen_ids:
                raise RunBundleLineageError("bundle lineage contains a cycle")
            current_id = predecessor_id
            del bundle
        else:
            raise RunBundleLineageError("bundle lineage exceeds configured depth")

        if root_manifest is None:
            raise RunBundleLineageError("bundle lineage has no root")
        validate_run_bundle_root(root_manifest, RunBundleLineageError)
        del bundle
        bundle_ids = tuple(reversed(reverse_bundle_ids))
        projection: ProjectionResult | None = None
        tip_bundle: RunBundleReader | None = None
        for exact_bundle_id in bundle_ids:
            bundle = self.source.read_exact(exact_bundle_id)
            if bundle is None:
                raise RunBundleLineageError(
                    f"bundle lineage changed during projection: {exact_bundle_id}"
                )
            if bundle.manifest.bundle_id != exact_bundle_id:
                raise RunBundleLineageError(
                    "exact bundle source changed identity during projection"
                )
            projection = self.projector.project(bundle, previous=projection)
            if exact_bundle_id == bundle_ids[-1]:
                tip_bundle = bundle
            else:
                del bundle
        if projection is None or tip_bundle is None:
            raise RunBundleLineageError("bundle lineage projection is empty")
        return VerifiedRunBundleLineage(
            bundle_ids=bundle_ids,
            tip_bundle=tip_bundle,
            tip_projection=projection,
        )
