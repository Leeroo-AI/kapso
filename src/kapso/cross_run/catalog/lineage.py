"""Exact immutable RunBundle lineage resolution and projection."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping, Protocol

from kapso.cross_run.canonical import require_content_id
from kapso.cross_run.capture.bundle import RunBundleReader
from kapso.cross_run.capture.bundle_lineage import (
    validate_run_bundle_root,
    validate_run_bundle_successor,
)
from kapso.cross_run.catalog.projector import ProjectionResult, RunBundleProjector
from kapso.cross_run.contracts import RunBundle


class RunBundleLineageError(ValueError):
    """An exact bundle lineage is absent, cyclic, or exceeds policy."""


class ExactRunBundleSource(Protocol):
    """Resolve only a caller-named immutable bundle identity."""

    def read_manifest_exact(
        self,
        bundle_id: str,
        *,
        deadline: float | None = None,
    ) -> RunBundle | None: ...

    def read_exact(self, bundle_id: str) -> RunBundleReader | None: ...

    def read_exact_bounded(
        self,
        bundle_id: str,
        *,
        maximum_entries: int,
        maximum_bytes: int,
        deadline: float,
    ) -> RunBundleReader | None: ...


@dataclass(frozen=True)
class VerifiedRunBundleLineage:
    """A root-to-tip bundle chain and its fully replayed tip projection."""

    bundles: tuple[RunBundleReader, ...]
    tip_projection: ProjectionResult

    def __post_init__(self) -> None:
        if not self.bundles:
            raise RunBundleLineageError("verified bundle lineage must not be empty")
        bundle_ids = self.bundle_ids
        for bundle_id in self.bundle_ids:
            require_content_id(bundle_id, "lineage bundle_id")
        if len(bundle_ids) != len(set(bundle_ids)):
            raise RunBundleLineageError("verified bundle lineage contains a cycle")
        validate_run_bundle_root(self.bundles[0].manifest, RunBundleLineageError)
        for parent, child in zip(self.bundles, self.bundles[1:]):
            validate_run_bundle_successor(
                parent.manifest,
                child.manifest,
                RunBundleLineageError,
            )
        if (
            bundle_ids[-1] != self.tip_bundle.manifest.bundle_id
            or self.tip_projection.source_bundle != self.tip_bundle.manifest
        ):
            raise RunBundleLineageError(
                "verified projection does not belong to the lineage tip"
            )

    @property
    def bundle_ids(self) -> tuple[str, ...]:
        return tuple(bundle.manifest.bundle_id for bundle in self.bundles)

    @property
    def tip_bundle(self) -> RunBundleReader:
        return self.bundles[-1]


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
        return self._resolve_exact(
            bundle_id,
            maximum_entries=None,
            maximum_bytes=None,
            deadline=None,
            retained_bundles={},
        )

    def resolve_exact_bounded(
        self,
        bundle_id: str,
        *,
        maximum_entries: int,
        maximum_bytes: int,
        timeout_seconds: int,
        retained_bundles: Mapping[str, RunBundleReader],
    ) -> VerifiedRunBundleLineage:
        if any(
            type(value) is not int or value <= 0
            for value in (maximum_entries, maximum_bytes, timeout_seconds)
        ):
            raise RunBundleLineageError(
                "bounded bundle lineage limits must be positive integers"
            )
        return self._resolve_exact(
            bundle_id,
            maximum_entries=maximum_entries,
            maximum_bytes=maximum_bytes,
            deadline=time.monotonic() + timeout_seconds,
            retained_bundles=retained_bundles,
        )

    def _resolve_exact(
        self,
        bundle_id: str,
        *,
        maximum_entries: int | None,
        maximum_bytes: int | None,
        deadline: float | None,
        retained_bundles: Mapping[str, RunBundleReader],
    ) -> VerifiedRunBundleLineage:
        require_content_id(bundle_id, "bundle_id")
        for retained_id, retained_bundle in retained_bundles.items():
            require_content_id(retained_id, "retained bundle_id")
            if retained_bundle.manifest.bundle_id != retained_id:
                raise RunBundleLineageError(
                    "retained bundle map contains another bundle identity"
                )
        retained_bundles_by_id = dict(retained_bundles)
        reverse_manifests: list[RunBundle] = []
        seen_ids: set[str] = set()
        current_id = bundle_id
        root_manifest: RunBundle | None = None
        while len(reverse_manifests) < self.maximum_bundles:
            self._require_deadline(deadline)
            if current_id in seen_ids:
                raise RunBundleLineageError("bundle lineage contains a cycle")
            seen_ids.add(current_id)
            retained_bundle = retained_bundles_by_id.get(current_id)
            manifest = (
                retained_bundle.manifest
                if retained_bundle is not None
                else self.source.read_manifest_exact(
                    current_id,
                    deadline=deadline,
                )
            )
            self._require_deadline(deadline)
            if manifest is None:
                raise RunBundleLineageError(
                    f"bundle lineage is missing exact predecessor: {current_id}"
                )
            if manifest.bundle_id != current_id:
                raise RunBundleLineageError(
                    "exact bundle source returned another bundle identity"
                )
            reverse_manifests.append(manifest)
            predecessor_id = manifest.supersedes_bundle_id
            if predecessor_id is None:
                root_manifest = manifest
                break
            require_content_id(predecessor_id, "supersedes_bundle_id")
            if predecessor_id in seen_ids:
                raise RunBundleLineageError("bundle lineage contains a cycle")
            current_id = predecessor_id
        else:
            raise RunBundleLineageError("bundle lineage exceeds configured depth")

        if root_manifest is None:
            raise RunBundleLineageError("bundle lineage has no root")
        validate_run_bundle_root(root_manifest, RunBundleLineageError)
        manifests = tuple(reversed(reverse_manifests))
        projection: ProjectionResult | None = None
        resolved_bundles: list[RunBundleReader] = []
        materialized_entries = 0
        materialized_bytes = 0
        for manifest in manifests:
            self._require_deadline(deadline)
            exact_bundle_id = manifest.bundle_id
            bundle = retained_bundles_by_id.get(exact_bundle_id)
            newly_materialized = bundle is None
            if bundle is None:
                if maximum_entries is None or maximum_bytes is None:
                    bundle = self.source.read_exact(exact_bundle_id)
                else:
                    remaining_entries = maximum_entries - materialized_entries
                    remaining_bytes = maximum_bytes - materialized_bytes
                    if remaining_entries <= 0 or remaining_bytes <= 0:
                        raise RunBundleLineageError(
                            "bundle lineage exceeds remaining replay materialization budget"
                        )
                    bundle = self.source.read_exact_bounded(
                        exact_bundle_id,
                        maximum_entries=remaining_entries,
                        maximum_bytes=remaining_bytes,
                        deadline=deadline,
                    )
            self._require_deadline(deadline)
            if bundle is None:
                raise RunBundleLineageError(
                    f"bundle lineage changed during projection: {exact_bundle_id}"
                )
            if bundle.manifest != manifest:
                raise RunBundleLineageError(
                    "exact bundle source changed manifest during projection"
                )
            if (
                newly_materialized
                and maximum_entries is not None
                and maximum_bytes is not None
            ):
                for relative_path in sorted(bundle.manifest.checksums):
                    self._require_deadline(deadline)
                    materialized_entries += 1
                    materialized_bytes += len(bundle.read_ref(relative_path))
                    if (
                        materialized_entries > maximum_entries
                        or materialized_bytes > maximum_bytes
                    ):
                        raise RunBundleLineageError(
                            "bundle lineage exceeds remaining replay materialization budget"
                        )
            projection = self.projector.project(bundle, previous=projection)
            self._require_deadline(deadline)
            resolved_bundles.append(bundle)
        if projection is None:
            raise RunBundleLineageError("bundle lineage projection is empty")
        return VerifiedRunBundleLineage(
            bundles=tuple(resolved_bundles),
            tip_projection=projection,
        )

    @staticmethod
    def _require_deadline(deadline: float | None) -> None:
        if deadline is not None and time.monotonic() >= deadline:
            raise RunBundleLineageError(
                "bundle lineage materialization deadline expired"
            )
