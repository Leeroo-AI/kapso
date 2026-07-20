"""Canonical JSON and content identity for cross-run artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from calendar import monthrange
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

CANONICALIZER_VERSION = "kapso.canonical_json.v1"
HASH_ALGORITHM = "sha256"

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
_CONTENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*:sha256:[0-9a-f]{64}$")
_UTC_TIMESTAMP_PATTERN = re.compile(
    r"^(?P<year>[0-9]{4})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})"
    r"T(?P<hour>[0-9]{2}):(?P<minute>[0-9]{2}):(?P<second>[0-9]{2})"
    r"(?:\.(?P<fraction>[0-9]{1,6}))?Z$"
)


class CanonicalizationError(ValueError):
    """Raised when input cannot participate in canonical identity."""


def require_identifier(value: Any, name: str) -> str:
    """Return a non-empty qualified identifier or fail loud."""
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise CanonicalizationError(f"{name} must be a qualified identifier")
    return value


def require_content_id(value: Any, name: str) -> str:
    """Return a structurally valid content identifier or fail loud."""
    if not isinstance(value, str) or not _CONTENT_ID_PATTERN.fullmatch(value):
        raise CanonicalizationError(
            f"{name} must have the form <namespace>:sha256:<64 lowercase hex>"
        )
    return value


def parse_utc_timestamp(value: Any, name: str) -> datetime:
    """Parse a timestamp only when its text is already normalized UTC."""
    if not isinstance(value, str):
        raise CanonicalizationError(f"{name} must be an ISO-8601 UTC timestamp")
    match = _UTC_TIMESTAMP_PATTERN.fullmatch(value)
    if match is None:
        raise CanonicalizationError(f"{name} must be an ISO-8601 UTC timestamp")
    parts = {
        key: int(match.group(key))
        for key in ("year", "month", "day", "hour", "minute", "second")
    }
    if not 1 <= parts["year"] <= 9999:
        raise CanonicalizationError(f"{name} has an invalid year")
    if not 1 <= parts["month"] <= 12:
        raise CanonicalizationError(f"{name} has an invalid month")
    if not 1 <= parts["day"] <= monthrange(parts["year"], parts["month"])[1]:
        raise CanonicalizationError(f"{name} has an invalid day")
    if not 0 <= parts["hour"] <= 23:
        raise CanonicalizationError(f"{name} has an invalid hour")
    if not 0 <= parts["minute"] <= 59:
        raise CanonicalizationError(f"{name} has an invalid minute")
    if not 0 <= parts["second"] <= 59:
        raise CanonicalizationError(f"{name} has an invalid second")
    fraction = match.group("fraction")
    microsecond = int(fraction.ljust(6, "0")) if fraction is not None else 0
    parsed = datetime(
        parts["year"],
        parts["month"],
        parts["day"],
        parts["hour"],
        parts["minute"],
        parts["second"],
        microsecond,
        timezone.utc,
    )
    timespec = "microseconds" if parsed.microsecond else "seconds"
    normalized = parsed.isoformat(timespec=timespec).replace("+00:00", "Z")
    if normalized != value:
        raise CanonicalizationError(f"{name} must be normalized as {normalized}")
    return parsed


def normalize_utc_timestamp(value: Any, name: str) -> str:
    """Validate that *value* is already a normalized UTC timestamp."""
    parse_utc_timestamp(value, name)
    return value


def _reject_duplicate_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalizationError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_non_finite_token(token: str) -> None:
    raise CanonicalizationError(f"non-finite JSON number: {token}")


def _parse_finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise CanonicalizationError(f"non-finite JSON number: {token}")
    return value


def parse_json_bytes(payload: bytes | str) -> Any:
    """Parse strict JSON, rejecting duplicate keys and non-finite numbers."""
    if isinstance(payload, bytes):
        text = payload.decode("utf-8")
    elif isinstance(payload, str):
        text = payload
    else:
        raise CanonicalizationError("JSON payload must be bytes or text")
    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_object_pairs,
        parse_constant=_reject_non_finite_token,
        parse_float=_parse_finite_float,
    )


def freeze_json(value: Any, name: str = "value") -> Any:
    """Validate and recursively freeze a JSON-compatible value."""
    if value is None or isinstance(value, str) or isinstance(value, bool):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise CanonicalizationError(f"{name} must be finite")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError(f"{name} object keys must be strings")
            if key in frozen:
                raise CanonicalizationError(f"{name} contains duplicate key {key}")
            frozen[key] = freeze_json(child, f"{name}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            freeze_json(child, f"{name}[{position}]")
            for position, child in enumerate(value)
        )
    raise CanonicalizationError(
        f"{name} has unsupported canonical type {type(value).__name__}"
    )


def to_json_value(value: Any) -> Any:
    """Convert records and frozen JSON containers to plain JSON values."""
    if value is None or isinstance(value, str) or isinstance(value, bool):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise CanonicalizationError("canonical numbers must be finite")
        return value
    if isinstance(value, Enum):
        return to_json_value(value.value)
    if isinstance(value, datetime):
        if value.tzinfo != timezone.utc:
            raise CanonicalizationError("canonical datetime must use UTC")
        timespec = "microseconds" if value.microsecond else "seconds"
        return value.isoformat(timespec=timespec).replace("+00:00", "Z")
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError("canonical object keys must be strings")
            normalized[key] = to_json_value(child)
        return normalized
    if isinstance(value, (list, tuple)):
        return [to_json_value(child) for child in value]
    if is_dataclass(value):
        return {
            field.name: to_json_value(getattr(value, field.name))
            for field in fields(value)
        }
    raise CanonicalizationError(f"unsupported canonical type {type(value).__name__}")


def canonical_json_bytes(payload: Any) -> bytes:
    """Return the sole canonical JSON byte representation."""
    normalized = to_json_value(payload)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def tree_or_blob_digest(payload: bytes) -> str:
    """Return an explicit SHA-256 digest for opaque bytes."""
    if not isinstance(payload, bytes):
        raise CanonicalizationError("digest input must be bytes")
    return f"{HASH_ALGORITHM}:{hashlib.sha256(payload).hexdigest()}"


def source_tree_digest(files: Mapping[str, tuple[str, str, int]]) -> str:
    """Hash an exact regular-file tree including paths, modes, sizes, and bytes."""
    if not files:
        raise CanonicalizationError("source tree descriptor must not be empty")
    descriptor = []
    for path in sorted(files):
        normalized = PurePosixPath(path)
        if (
            normalized.is_absolute()
            or ".." in normalized.parts
            or normalized.as_posix() != path
            or path == "."
        ):
            raise CanonicalizationError(
                "source tree path must be normalized and relative"
            )
        digest, mode, size = files[path]
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise CanonicalizationError("source tree file digest is invalid")
        if mode not in {"100644", "100755"}:
            raise CanonicalizationError("source tree file mode is invalid")
        if type(size) is not int or size < 0:
            raise CanonicalizationError("source tree file size is invalid")
        descriptor.append({"digest": digest, "mode": mode, "path": path, "size": size})
    return tree_or_blob_digest(canonical_json_bytes(tuple(descriptor)))


def content_id(namespace: str, payload_without_id: Any) -> str:
    """Mint a namespaced content ID from a canonical immutable payload."""
    require_identifier(namespace, "content namespace")
    if ":" in namespace or "/" in namespace:
        raise CanonicalizationError("content namespace cannot contain ':' or '/'")
    digest = hashlib.sha256(canonical_json_bytes(payload_without_id)).hexdigest()
    return f"{namespace}:{HASH_ALGORITHM}:{digest}"


def verify_declared_content_id(
    payload: Mapping[str, Any],
    *,
    namespace: str,
    identity_field: str,
    excluded_fields: tuple[str, ...] = (),
) -> None:
    """Verify an object's declared ID against its immutable preimage."""
    if identity_field not in payload:
        raise CanonicalizationError(f"missing identity field: {identity_field}")
    declared = require_content_id(payload[identity_field], identity_field)
    excluded = set(excluded_fields) | {identity_field}
    preimage = {key: value for key, value in payload.items() if key not in excluded}
    expected = content_id(namespace, preimage)
    if declared != expected:
        raise CanonicalizationError(
            f"{identity_field} mismatch: declared {declared}, expected {expected}"
        )
