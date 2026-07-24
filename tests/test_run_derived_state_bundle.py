from __future__ import annotations

from io import BytesIO

import pytest

from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.launch.derived_state_bundle import (
    RunDerivedStateBundle,
    RunDerivedStateBundleError,
    generation_object_name,
)
from kapso.cross_run.launch.derived_state_contracts import (
    RunDerivedStateGeneration,
    RunStateAuthority,
    RunStateLayout,
    RunStatePayloadTransition,
)


def _identifier(namespace: str, value: str) -> str:
    return content_id(namespace, {"value": value})


def _bundle() -> RunDerivedStateBundle:
    payload_by_authority = {
        RunStateAuthority.IDEA_ARCHIVE: b'{"revision":0}',
        RunStateAuthority.EXECUTION_JOURNAL: (
            b'{"event":"one","unicode":"\xe2\x88\x86"}\n'
        ),
        RunStateAuthority.EXPERIMENT_HISTORY: b'{"records":[],"revision":0}',
    }
    layout = RunStateLayout.build(
        strategy_kind="generic",
        authority_paths={
            RunStateAuthority.IDEA_ARCHIVE: ".kapso/idea_archive.json",
            RunStateAuthority.EXPERIMENT_HISTORY: (".kapso/experiment_history.json"),
            RunStateAuthority.EXECUTION_JOURNAL: ".kapso/execution_events.jsonl",
        },
    )
    transitions = tuple(
        RunStatePayloadTransition.mint(
            authority_binding_id=binding.authority_binding_id,
            predecessor_digest=None,
            predecessor_revision=None,
            predecessor_size_bytes=None,
            target_digest=tree_or_blob_digest(payload_by_authority[binding.authority]),
            target_revision=0,
            target_size_bytes=len(payload_by_authority[binding.authority]),
        )
        for binding in layout.bindings
    )
    generation = RunDerivedStateGeneration.build(
        bootstrap_pin_id=_identifier("bootstrap-pin", "bootstrap"),
        run_state_layout=layout,
        predecessor_checkpoint_head_id=_identifier(
            "run-checkpoint-head",
            "head",
        ),
        predecessor_checkpoint_id=None,
        predecessor_evidence_id=None,
        target_evidence_id=_identifier("run-derivative-evidence", "evidence"),
        payload_transitions=transitions,
    )
    return RunDerivedStateBundle(
        generation=generation,
        payloads=tuple(
            payload_by_authority[binding.authority] for binding in layout.bindings
        ),
    )


def _generation_with_payload(
    bundle: RunDerivedStateBundle,
    position: int,
    payload: bytes,
) -> RunDerivedStateGeneration:
    transitions = list(bundle.generation.payload_transitions)
    current = transitions[position]
    transitions[position] = RunStatePayloadTransition.mint(
        authority_binding_id=current.authority_binding_id,
        predecessor_digest=current.predecessor_digest,
        predecessor_revision=current.predecessor_revision,
        predecessor_size_bytes=current.predecessor_size_bytes,
        target_digest=tree_or_blob_digest(payload),
        target_revision=current.target_revision,
        target_size_bytes=len(payload),
    )
    return RunDerivedStateGeneration.build(
        bootstrap_pin_id=bundle.generation.bootstrap_pin_id,
        run_state_layout=bundle.generation.run_state_layout,
        predecessor_checkpoint_head_id=(
            bundle.generation.predecessor_checkpoint_head_id
        ),
        predecessor_checkpoint_id=bundle.generation.predecessor_checkpoint_id,
        predecessor_evidence_id=bundle.generation.predecessor_evidence_id,
        target_evidence_id=bundle.generation.target_evidence_id,
        payload_transitions=tuple(transitions),
    )


def test_bundle_round_trips_exact_raw_payloads() -> None:
    bundle = _bundle()

    restored = RunDerivedStateBundle.from_bytes(bundle.to_bytes())

    assert restored == bundle
    assert restored.payload_by_relative_path() == bundle.payload_by_relative_path()
    assert restored.to_bytes() == bundle.to_bytes()
    assert b"".join(restored.iter_bytes()) == bundle.to_bytes()
    assert b"\xe2\x88\x86" in restored.to_bytes()


def test_bundle_stream_reader_preserves_exact_payloads_without_full_copy() -> None:
    bundle = _bundle()

    restored = RunDerivedStateBundle.read_from(
        BytesIO(bundle.to_bytes()),
        maximum_bytes=bundle.byte_size,
    )

    assert restored == bundle
    assert restored.byte_size == len(bundle.to_bytes())
    assert restored.digest == tree_or_blob_digest(bundle.to_bytes())


def test_bundle_stream_reader_enforces_bound_and_complete_eof() -> None:
    payload = _bundle().to_bytes()

    with pytest.raises(RunDerivedStateBundleError, match="bound"):
        RunDerivedStateBundle.read_from(
            BytesIO(payload),
            maximum_bytes=len(payload) - 1,
        )
    with pytest.raises(RunDerivedStateBundleError, match="trailing"):
        RunDerivedStateBundle.read_from(
            BytesIO(payload + b"x"),
            maximum_bytes=len(payload) + 1,
        )


def test_bundle_object_name_is_generation_identity() -> None:
    bundle = _bundle()

    assert bundle.object_name == (
        f"generation-{bundle.generation.generation_id.rsplit(':', 1)[1]}.bundle"
    )
    assert generation_object_name(bundle.generation.generation_id) == (
        bundle.object_name
    )


def test_bundle_rejects_payload_digest_substitution() -> None:
    bundle = _bundle()
    substituted = list(bundle.payloads)
    substituted[0] = b'{"revision":1}'

    with pytest.raises(
        RunDerivedStateBundleError,
        match="differs from its transition",
    ):
        RunDerivedStateBundle(
            generation=bundle.generation,
            payloads=tuple(substituted),
        )


def test_bundle_rejects_noncanonical_authority_json() -> None:
    bundle = _bundle()
    archive_position = tuple(
        binding.authority for binding in bundle.generation.run_state_layout.bindings
    ).index(RunStateAuthority.IDEA_ARCHIVE)
    payload = b'{ "revision": 0 }'
    generation = _generation_with_payload(bundle, archive_position, payload)
    payloads = list(bundle.payloads)
    payloads[archive_position] = payload

    with pytest.raises(RunDerivedStateBundleError, match="not one canonical"):
        RunDerivedStateBundle(
            generation=generation,
            payloads=tuple(payloads),
        )


def test_bundle_rejects_noncanonical_jsonl_tail() -> None:
    bundle = _bundle()
    journal_position = tuple(
        binding.authority for binding in bundle.generation.run_state_layout.bindings
    ).index(RunStateAuthority.EXECUTION_JOURNAL)
    payload = b'{"event":"one"}'
    generation = _generation_with_payload(bundle, journal_position, payload)
    payloads = list(bundle.payloads)
    payloads[journal_position] = payload

    with pytest.raises(RunDerivedStateBundleError, match="incomplete tail"):
        RunDerivedStateBundle(
            generation=generation,
            payloads=tuple(payloads),
        )


@pytest.mark.parametrize("suffix", [b"", b"unexpected"])
def test_bundle_decoder_rejects_truncation_and_trailing_bytes(suffix: bytes) -> None:
    encoded = _bundle().to_bytes()
    candidate = encoded[:-1] if not suffix else encoded + suffix

    with pytest.raises(RunDerivedStateBundleError):
        RunDerivedStateBundle.from_bytes(candidate)


def test_bundle_decoder_rejects_manifest_length_substitution() -> None:
    encoded = _bundle().to_bytes()
    magic_size = len(b"KAPSO_RUN_DERIVED_STATE_BUNDLE_V1\n")
    declared = int.from_bytes(encoded[magic_size : magic_size + 8], "big")
    substituted = (
        encoded[:magic_size]
        + (declared + 1).to_bytes(8, "big")
        + encoded[magic_size + 8 :]
    )

    with pytest.raises((RunDerivedStateBundleError, ValueError)):
        RunDerivedStateBundle.from_bytes(substituted)


def test_generation_object_name_rejects_another_namespace() -> None:
    with pytest.raises(RunDerivedStateBundleError, match="wrong namespace"):
        generation_object_name(_identifier("another-generation", "generation"))
