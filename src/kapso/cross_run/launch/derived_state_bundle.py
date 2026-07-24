"""Compact retained bundles for one checkpoint-governed run-state generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from collections.abc import Iterator
from typing import BinaryIO, Mapping

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    parse_json_bytes,
    require_content_id,
    tree_or_blob_digest,
)
from kapso.cross_run.launch.derived_state_contracts import (
    RunDerivedStateGeneration,
    RunStatePayloadFormat,
)

_BUNDLE_MAGIC = b"KAPSO_RUN_DERIVED_STATE_BUNDLE_V1\n"
_MANIFEST_LENGTH_BYTES = 8


class RunDerivedStateBundleError(ValueError):
    """A retained run-state generation bundle is incomplete or noncanonical."""


@dataclass(frozen=True)
class RunDerivedStateBundle:
    """One generation manifest followed by its exact raw authority payloads."""

    generation: RunDerivedStateGeneration
    payloads: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if type(self.generation) is not RunDerivedStateGeneration:
            raise RunDerivedStateBundleError(
                "run derived-state bundle requires one exact generation"
            )
        if type(self.payloads) is not tuple or any(
            type(payload) is not bytes for payload in self.payloads
        ):
            raise RunDerivedStateBundleError(
                "run derived-state bundle payloads must be an immutable byte tuple"
            )
        bindings = self.generation.run_state_layout.bindings
        transitions = self.generation.payload_transitions
        if len(self.payloads) != len(bindings):
            raise RunDerivedStateBundleError(
                "run derived-state bundle payload set is incomplete"
            )
        for binding, transition, payload in zip(
            bindings,
            transitions,
            self.payloads,
            strict=True,
        ):
            if (
                transition.authority_binding_id != binding.authority_binding_id
                or transition.target_size_bytes != len(payload)
                or transition.target_digest != tree_or_blob_digest(payload)
            ):
                raise RunDerivedStateBundleError(
                    "run derived-state payload differs from its transition"
                )
            _require_canonical_payload(payload, binding.payload_format)

    @property
    def object_name(self) -> str:
        """Return the sole store filename for this generation."""
        return generation_object_name(self.generation.generation_id)

    def payload_by_relative_path(self) -> Mapping[str, bytes]:
        """Return the complete path-keyed set of repairable view bytes."""
        return {
            binding.relative_path: payload
            for binding, payload in zip(
                self.generation.run_state_layout.bindings,
                self.payloads,
                strict=True,
            )
        }

    @property
    def byte_size(self) -> int:
        """Return the exact framed size without materializing another full copy."""
        return sum(len(chunk) for chunk in self.iter_bytes())

    @property
    def digest(self) -> str:
        """Hash the exact framed bytes without materializing another full copy."""
        hasher = hashlib.sha256()
        for chunk in self.iter_bytes():
            hasher.update(chunk)
        return f"sha256:{hasher.hexdigest()}"

    def to_bytes(self) -> bytes:
        """Materialize the exact bundle bytes for bounded in-memory consumers."""
        return b"".join(self.iter_bytes())

    def iter_bytes(self) -> Iterator[bytes]:
        """Yield framing and raw payloads without copying the retained authorities."""
        manifest = self.generation.to_json_bytes()
        yield _BUNDLE_MAGIC
        yield len(manifest).to_bytes(_MANIFEST_LENGTH_BYTES, "big")
        yield manifest
        yield from self.payloads

    @classmethod
    def from_bytes(cls, payload: bytes) -> "RunDerivedStateBundle":
        """Decode only the exact, complete retained-bundle representation."""
        if type(payload) is not bytes:
            raise RunDerivedStateBundleError(
                "run derived-state bundle payload must be bytes"
            )
        header_size = len(_BUNDLE_MAGIC) + _MANIFEST_LENGTH_BYTES
        if len(payload) < header_size or not payload.startswith(_BUNDLE_MAGIC):
            raise RunDerivedStateBundleError(
                "run derived-state bundle header is invalid"
            )
        manifest_size = int.from_bytes(
            payload[len(_BUNDLE_MAGIC) : header_size],
            "big",
        )
        manifest_end = header_size + manifest_size
        if manifest_size == 0 or manifest_end > len(payload):
            raise RunDerivedStateBundleError(
                "run derived-state bundle manifest length is invalid"
            )
        manifest_payload = payload[header_size:manifest_end]
        generation = RunDerivedStateGeneration.from_json_bytes(manifest_payload)
        if generation.to_json_bytes() != manifest_payload:
            raise RunDerivedStateBundleError(
                "run derived-state bundle manifest is not canonical"
            )
        position = manifest_end
        decoded_payloads: list[bytes] = []
        for transition in generation.payload_transitions:
            end = position + transition.target_size_bytes
            if end > len(payload):
                raise RunDerivedStateBundleError(
                    "run derived-state bundle contains a truncated authority payload"
                )
            decoded_payloads.append(payload[position:end])
            position = end
        if position != len(payload):
            raise RunDerivedStateBundleError(
                "run derived-state bundle contains trailing bytes"
            )
        bundle = cls(
            generation=generation,
            payloads=tuple(decoded_payloads),
        )
        if bundle.to_bytes() != payload:
            raise RunDerivedStateBundleError(
                "run derived-state bundle bytes are not canonical"
            )
        return bundle

    @classmethod
    def read_from(
        cls,
        handle: BinaryIO,
        *,
        maximum_bytes: int,
    ) -> "RunDerivedStateBundle":
        """Decode one bounded regular-file stream without duplicating its payload."""
        if (
            not hasattr(handle, "read")
            or type(maximum_bytes) is not int
            or maximum_bytes <= 0
        ):
            raise RunDerivedStateBundleError(
                "run derived-state bundle reader is invalid"
            )
        prefix = handle.read(len(_BUNDLE_MAGIC) + _MANIFEST_LENGTH_BYTES)
        if len(prefix) != len(
            _BUNDLE_MAGIC
        ) + _MANIFEST_LENGTH_BYTES or not prefix.startswith(_BUNDLE_MAGIC):
            raise RunDerivedStateBundleError(
                "run derived-state bundle header is invalid"
            )
        manifest_size = int.from_bytes(
            prefix[len(_BUNDLE_MAGIC) :],
            "big",
        )
        if manifest_size == 0 or len(prefix) + manifest_size > maximum_bytes:
            raise RunDerivedStateBundleError(
                "run derived-state bundle manifest length is invalid"
            )
        manifest_payload = handle.read(manifest_size)
        if len(manifest_payload) != manifest_size:
            raise RunDerivedStateBundleError(
                "run derived-state bundle manifest is truncated"
            )
        generation = RunDerivedStateGeneration.from_json_bytes(manifest_payload)
        if generation.to_json_bytes() != manifest_payload:
            raise RunDerivedStateBundleError(
                "run derived-state bundle manifest is not canonical"
            )
        total_size = len(prefix) + manifest_size
        decoded_payloads = []
        for transition in generation.payload_transitions:
            total_size += transition.target_size_bytes
            if total_size > maximum_bytes:
                raise RunDerivedStateBundleError(
                    "run derived-state bundle exceeds its configured bound"
                )
            payload = handle.read(transition.target_size_bytes)
            if len(payload) != transition.target_size_bytes:
                raise RunDerivedStateBundleError(
                    "run derived-state bundle contains a truncated authority payload"
                )
            decoded_payloads.append(payload)
        if handle.read(1):
            raise RunDerivedStateBundleError(
                "run derived-state bundle contains trailing bytes"
            )
        bundle = cls(
            generation=generation,
            payloads=tuple(decoded_payloads),
        )
        if bundle.byte_size != total_size:
            raise RunDerivedStateBundleError(
                "run derived-state bundle bytes are not canonical"
            )
        return bundle


def generation_object_name(generation_id: str) -> str:
    """Map one exact generation identity to its permanent object filename."""
    require_content_id(generation_id, "run derived-state generation")
    if (
        generation_id.split(":sha256:", 1)[0]
        != RunDerivedStateGeneration.CONTENT_NAMESPACE
    ):
        raise RunDerivedStateBundleError(
            "run derived-state generation uses the wrong namespace"
        )
    return f"generation-{generation_id.rsplit(':', 1)[1]}.bundle"


def _require_canonical_payload(
    payload: bytes,
    payload_format: RunStatePayloadFormat,
) -> None:
    if payload_format is RunStatePayloadFormat.CANONICAL_JSON:
        parsed = parse_json_bytes(payload)
        if not isinstance(parsed, Mapping) or canonical_json_bytes(parsed) != payload:
            raise RunDerivedStateBundleError(
                "run derived-state JSON payload is not one canonical object"
            )
        return
    if payload_format is not RunStatePayloadFormat.CANONICAL_JSONL:
        raise RunDerivedStateBundleError(
            "run derived-state payload format is unsupported"
        )
    if not payload:
        return
    if not payload.endswith(b"\n"):
        raise RunDerivedStateBundleError(
            "run derived-state JSONL payload has an incomplete tail"
        )
    lines = payload.split(b"\n")[:-1]
    if any(not line for line in lines):
        raise RunDerivedStateBundleError(
            "run derived-state JSONL payload contains a blank record"
        )
    for line in lines:
        parsed = parse_json_bytes(line)
        if not isinstance(parsed, Mapping) or canonical_json_bytes(parsed) != line:
            raise RunDerivedStateBundleError(
                "run derived-state JSONL record is not one canonical object"
            )


__all__ = [
    "RunDerivedStateBundle",
    "RunDerivedStateBundleError",
    "generation_object_name",
]
