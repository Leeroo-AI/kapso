"""Dependency-light deterministic sanitation rules shared by artifact gates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from kapso.cross_run.canonical import canonical_json_bytes, tree_or_blob_digest
from kapso.cross_run.capture.safety import path_matches_denied_pattern
from kapso.cross_run.settings import SanitationSettings

SANITATION_SCANNER_VERSION = "kapso.deterministic_text_scanner.v1"

_SPDX_PATTERN = re.compile(r"SPDX-License-Identifier:\s*([^\s*]+)")
_SECRET_PATTERNS = (
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("github_token", re.compile(r"\bgh(?:p|o|u|s|r)_[A-Za-z0-9]{20,}\b")),
    ("openai_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    (
        "assigned_secret",
        re.compile(
            r"(?im)^\s*(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)"
            r"\s*[:=]\s*['\"]?[^\s'\"${}]{8,}"
        ),
    ),
    ("credential_url", re.compile(r"https?://[^\s/:]+:[^\s/@]+@")),
)


@dataclass(frozen=True)
class SanitationRuleFinding:
    code: str
    relative_path: str
    evidence_digest: str
    severity: str


def sanitation_policy_fingerprint(settings: SanitationSettings) -> str:
    """Bind a report to every effective deterministic sanitation setting."""
    return tree_or_blob_digest(canonical_json_bytes(settings.to_dict()))


def scan_text_artifact(
    settings: SanitationSettings,
    relative_path: str,
    payload: bytes,
) -> tuple[SanitationRuleFinding, ...]:
    """Return canonical findings for one candidate publication artifact."""
    path = PurePosixPath(relative_path)
    source_name = path.name
    suffix = path.suffix.casefold()
    evidence_digest = tree_or_blob_digest(payload)
    findings: set[tuple[str, str]] = set()
    if path_matches_denied_pattern(relative_path, settings.denied_path_patterns):
        findings.add(("denied_path", "reject"))
    if (
        suffix not in settings.allowed_suffixes
        and source_name not in settings.allowed_filenames
    ):
        findings.add(("forbidden_artifact_class", "reject"))
    if len(payload) > settings.max_file_bytes:
        findings.add(("file_too_large", "reject"))
    text = payload.decode("utf-8", errors="surrogateescape")
    if any(0xDC80 <= ord(character) <= 0xDCFF for character in text):
        findings.add(("invalid_utf8", "reject"))
    if "\x00" in text:
        findings.add(("nul_byte", "reject"))
    for code, pattern in _SECRET_PATTERNS:
        if pattern.search(text) is not None:
            findings.add((code, "reject"))
    spdx_licenses = tuple(sorted(set(_SPDX_PATTERN.findall(text))))
    for license_id in spdx_licenses:
        if license_id not in settings.allowed_spdx_licenses:
            findings.add(("unapproved_spdx_license", "reject"))
    if source_name == "LICENSE" and not spdx_licenses:
        findings.add(("unclassified_license", "notice"))
    return tuple(
        SanitationRuleFinding(
            code=code,
            relative_path=relative_path,
            evidence_digest=evidence_digest,
            severity=severity,
        )
        for code, severity in sorted(findings)
    )
