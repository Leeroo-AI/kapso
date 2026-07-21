"""Deterministic sanitation authority for exact expert candidate trees."""

from __future__ import annotations

from typing import Mapping

from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.contracts import (
    ExpertCandidateSanitationFinding,
    ExpertCandidateSanitationReport,
    ExpertCandidateSanitationStatus,
    ExpertSanitationSeverity,
    ExpertSourceTreeManifest,
)
from kapso.cross_run.sanitation_rules import (
    SANITATION_SCANNER_VERSION,
    sanitation_policy_fingerprint,
    scan_text_artifact,
)
from kapso.cross_run.settings import SanitationSettings


class ExpertCandidateSanitizer:
    """Recompute admission from configured policy and exact candidate bytes."""

    def __init__(self, settings: SanitationSettings):
        self.settings = settings

    def scan(
        self,
        scope_contract_id: str,
        candidate_tree: ExpertSourceTreeManifest,
        candidate_contents: Mapping[str, bytes],
    ) -> ExpertCandidateSanitationReport:
        files = {file.relative_path: file for file in candidate_tree.files}
        if set(candidate_contents) != set(files):
            raise ValueError("sanitation bytes differ from candidate tree paths")
        findings: list[ExpertCandidateSanitationFinding] = []
        for path in sorted(files):
            payload = candidate_contents[path]
            descriptor = files[path]
            if not isinstance(payload, bytes):
                raise ValueError("sanitation candidate content must be bytes")
            if (
                tree_or_blob_digest(payload) != descriptor.digest
                or len(payload) != descriptor.size
            ):
                raise ValueError(
                    f"sanitation bytes differ from candidate descriptor: {path}"
                )
            for finding in scan_text_artifact(self.settings, path, payload):
                findings.append(
                    ExpertCandidateSanitationFinding(
                        code=finding.code,
                        relative_path=path,
                        evidence_digest=finding.evidence_digest,
                        severity=(
                            ExpertSanitationSeverity.BLOCKING
                            if finding.severity == "reject"
                            else ExpertSanitationSeverity.WARNING
                        ),
                    )
                )
        rejected = any(
            finding.severity is ExpertSanitationSeverity.BLOCKING
            for finding in findings
        )
        return ExpertCandidateSanitationReport.mint(
            scope_contract_id=scope_contract_id,
            candidate_tree_hash=candidate_tree.tree_hash,
            policy_version=self.settings.policy_version,
            policy_fingerprint=sanitation_policy_fingerprint(self.settings),
            scanner_version=SANITATION_SCANNER_VERSION,
            status=(
                ExpertCandidateSanitationStatus.REJECTED
                if rejected
                else ExpertCandidateSanitationStatus.ADMITTED
            ),
            scanned_files=candidate_tree.files,
            findings=tuple(findings),
        )
