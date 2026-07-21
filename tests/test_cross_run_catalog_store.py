from __future__ import annotations

import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.catalog.store import (
    CatalogClosureError,
    CatalogCompareAndSwapError,
    CatalogCorruptionError,
    CatalogGenerationManifest,
    CatalogInputDelta,
    CatalogLayoutError,
    CatalogNotInitializedError,
    CatalogOperationConflictError,
    CatalogReducerError,
    CatalogReduction,
    CatalogReductionRequest,
    CatalogStore,
)
from kapso.cross_run.contracts import (
    AdmissionState,
    CatalogEntryState,
    ContractValidationError,
    StrictContract,
)

SCOPE_CONTRACT_ID = content_id("test-scope", {"scope": "catalog-store"})
CONFIGURATION_FINGERPRINT = tree_or_blob_digest(b"catalog-test-config")


@dataclass(frozen=True)
class CatalogStoreFact(StrictContract):
    fact_id: str
    fact_kind: str
    logical_key: str
    ordinal: int

    CONTENT_NAMESPACE = "test-catalog-fact"
    IDENTITY_FIELD = "fact_id"


@dataclass(frozen=True)
class AttestedCatalogStoreFact(StrictContract):
    fact_id: str
    value: str
    attestation: dict[str, Any]

    CONTENT_NAMESPACE = "test-attested-catalog-fact"
    IDENTITY_FIELD = "fact_id"
    CONTENT_EXCLUDED_FIELDS = ("attestation",)


class InjectedCatalogCrash(RuntimeError):
    pass


def fact(kind: str, logical_key: str, ordinal: int) -> CatalogStoreFact:
    return CatalogStoreFact.mint(
        fact_kind=kind,
        logical_key=logical_key,
        ordinal=ordinal,
    )


def input_delta(
    operation_id: str,
    objects: tuple[CatalogStoreFact, ...],
    *,
    dependency_ids: tuple[str, ...] = (),
    configuration_fingerprint: str = CONFIGURATION_FINGERPRINT,
) -> CatalogInputDelta:
    object_ids = tuple(sorted(record.fact_id for record in objects))
    return CatalogInputDelta.mint(
        scope_contract_id=SCOPE_CONTRACT_ID,
        operation_id=operation_id,
        configuration_fingerprint=configuration_fingerprint,
        added_object_ids=object_ids,
        dependency_closure_ids=tuple(sorted(set(object_ids) | set(dependency_ids))),
    )


def reduce_complete_facts(request: CatalogReductionRequest) -> CatalogReduction:
    bundles: dict[str, CatalogStoreFact] = {}
    states: dict[str, CatalogStoreFact] = {}
    for object_id in request.fact_object_ids:
        record = CatalogStoreFact.from_json_bytes(request.read_object_bytes(object_id))
        destination = bundles if record.fact_kind == "bundle" else states
        incumbent = destination.get(record.logical_key)
        if incumbent is None or (record.ordinal, record.fact_id) > (
            incumbent.ordinal,
            incumbent.fact_id,
        ):
            destination[record.logical_key] = record
    return CatalogReduction(
        bundle_frontier={key: record.fact_id for key, record in bundles.items()},
        active_entry_state_ids={key: record.fact_id for key, record in states.items()},
        derived_objects=(),
    )


def reduce_without_derived_references(
    request: CatalogReductionRequest,
) -> CatalogReduction:
    return CatalogReduction(
        bundle_frontier={},
        active_entry_state_ids={},
        derived_objects=(),
    )


def reduce_catalog_entry_states(
    request: CatalogReductionRequest,
) -> CatalogReduction:
    states = tuple(
        CatalogEntryState.mint(
            subject_payload_id=subject_id,
            catalog_generation=request.generation_number,
            predecessor_state_id=request.parent_generation.active_entry_state_ids.get(
                subject_id
            ),
            configuration_fingerprint=request.configuration_fingerprint,
            admission_state=AdmissionState.QUARANTINED,
            superseded_by_payload_ids=(),
            assertion_ids=(),
            revocation_ids=(),
            taint_source_ids=(),
        )
        for subject_id in request.fact_object_ids
    )
    return CatalogReduction(
        bundle_frontier={},
        active_entry_state_ids={
            state.subject_payload_id: state.catalog_entry_state_id for state in states
        },
        derived_objects=states,
    )


def reduce_states_for_previous_generation(
    request: CatalogReductionRequest,
) -> CatalogReduction:
    states = tuple(
        CatalogEntryState.mint(
            subject_payload_id=subject_id,
            catalog_generation=request.parent_generation.generation_number,
            predecessor_state_id=request.parent_generation.active_entry_state_ids.get(
                subject_id
            ),
            configuration_fingerprint=request.configuration_fingerprint,
            admission_state=AdmissionState.QUARANTINED,
            superseded_by_payload_ids=(),
            assertion_ids=(),
            revocation_ids=(),
            taint_source_ids=(),
        )
        for subject_id in request.fact_object_ids
    )
    return CatalogReduction(
        bundle_frontier={},
        active_entry_state_ids={
            state.subject_payload_id: state.catalog_entry_state_id for state in states
        },
        derived_objects=states,
    )


def publish_rebased_in_process(
    root: str,
    delta: CatalogInputDelta,
    record: CatalogStoreFact,
    start: multiprocessing.synchronize.Event,
    completed: multiprocessing.queues.Queue,
) -> None:
    store = CatalogStore(root)
    start.wait()
    result = store.rebase(
        input_delta=delta,
        objects=(record,),
        reducer=reduce_complete_facts,
    )
    completed.put(result.generation.catalog_generation_id)


def publish_expected_in_process(
    root: str,
    expected: CatalogGenerationManifest,
    delta: CatalogInputDelta,
    record: CatalogStoreFact,
    start: multiprocessing.synchronize.Event,
    completed: multiprocessing.queues.Queue,
) -> None:
    store = CatalogStore(root)
    start.wait()
    result = store.publish(
        expected_generation_id=expected.catalog_generation_id,
        expected_generation_number=expected.generation_number,
        input_delta=delta,
        objects=(record,),
        reducer=reduce_complete_facts,
    )
    completed.put(result.generation.catalog_generation_id)


def initialized_store(
    root: Path,
) -> tuple[CatalogStore, CatalogGenerationManifest]:
    store = CatalogStore(root)
    generation = store.initialize(
        scope_contract_id=SCOPE_CONTRACT_ID,
        configuration_fingerprint=CONFIGURATION_FINGERPRINT,
    )
    return store, generation


def test_generation_zero_is_canonical_empty_and_initialization_is_idempotent(
    tmp_path: Path,
) -> None:
    store, initial = initialized_store(tmp_path / "catalog")

    repeated = store.initialize(
        scope_contract_id=SCOPE_CONTRACT_ID,
        configuration_fingerprint=CONFIGURATION_FINGERPRINT,
    )

    assert repeated == initial
    assert initial.generation_number == 0
    assert initial.parent_generation_id is None
    assert initial.fact_object_ids == ()
    assert initial.derived_object_ids == ()
    assert initial.applied_input_delta_ids == ()
    assert dict(initial.bundle_frontier) == {}
    assert dict(initial.active_entry_state_ids) == {}
    assert store.read_current() == initial


def test_publish_and_stale_rebase_union_facts_then_rerun_the_complete_reducer(
    tmp_path: Path,
) -> None:
    store, initial = initialized_store(tmp_path / "catalog")
    first = fact("bundle", "run-a", 1)
    first_delta = input_delta("capture_run_a", (first,))
    first_commit = store.publish(
        expected_generation_id=initial.catalog_generation_id,
        expected_generation_number=initial.generation_number,
        input_delta=first_delta,
        objects=(first,),
        reducer=reduce_complete_facts,
    )
    second = fact("bundle", "run-b", 1)
    second_delta = input_delta("capture_run_b", (second,))

    with pytest.raises(CatalogCompareAndSwapError):
        store.publish(
            expected_generation_id=initial.catalog_generation_id,
            expected_generation_number=initial.generation_number,
            input_delta=second_delta,
            objects=(second,),
            reducer=reduce_complete_facts,
        )
    rebased = store.rebase(
        input_delta=second_delta,
        objects=(second,),
        reducer=reduce_complete_facts,
    )

    assert rebased.generation.generation_number == 2
    assert rebased.generation.parent_generation_id == (
        first_commit.generation.catalog_generation_id
    )
    assert rebased.generation.fact_object_ids == tuple(
        sorted((first.fact_id, second.fact_id))
    )
    assert dict(rebased.generation.bundle_frontier) == {
        "run-a": first.fact_id,
        "run-b": second.fact_id,
    }
    assert rebased.delta_manifest is not None
    assert rebased.delta_manifest.added_object_ids == (second.fact_id,)
    assert dict(rebased.delta_manifest.bundle_frontier_changes) == {
        "run-b": second.fact_id
    }


def test_stale_compare_and_swap_writes_no_objects_or_operation_binding(
    tmp_path: Path,
) -> None:
    store, initial = initialized_store(tmp_path / "catalog")
    winner = fact("bundle", "winner", 1)
    store.publish(
        expected_generation_id=initial.catalog_generation_id,
        expected_generation_number=0,
        input_delta=input_delta("winner_operation", (winner,)),
        objects=(winner,),
        reducer=reduce_complete_facts,
    )
    object_names_before = tuple(
        sorted(path.name for path in store.objects_path.iterdir())
    )
    operation_names_before = tuple(
        sorted(path.name for path in store.operations_path.iterdir())
    )
    loser = fact("bundle", "loser", 1)

    with pytest.raises(CatalogCompareAndSwapError):
        store.publish(
            expected_generation_id=initial.catalog_generation_id,
            expected_generation_number=0,
            input_delta=input_delta("loser_operation", (loser,)),
            objects=(loser,),
            reducer=reduce_complete_facts,
        )

    assert tuple(sorted(path.name for path in store.objects_path.iterdir())) == (
        object_names_before
    )
    assert tuple(sorted(path.name for path in store.operations_path.iterdir())) == (
        operation_names_before
    )


def test_exact_operation_replay_is_idempotent_even_with_stale_expectation(
    tmp_path: Path,
) -> None:
    store, initial = initialized_store(tmp_path / "catalog")
    record = fact("bundle", "run-a", 1)
    delta = input_delta("same_operation", (record,))
    first = store.publish(
        expected_generation_id=initial.catalog_generation_id,
        expected_generation_number=0,
        input_delta=delta,
        objects=(record,),
        reducer=reduce_complete_facts,
    )

    replay = store.publish(
        expected_generation_id=initial.catalog_generation_id,
        expected_generation_number=0,
        input_delta=delta,
        objects=(record,),
        reducer=reduce_complete_facts,
    )

    assert replay.replayed
    assert replay.delta_manifest is None
    assert replay.generation == first.generation
    assert store.read_current().generation_number == 1


def test_conflicting_operation_reuse_fails_before_persisting_new_facts(
    tmp_path: Path,
) -> None:
    store, initial = initialized_store(tmp_path / "catalog")
    first = fact("bundle", "run-a", 1)
    store.publish(
        expected_generation_id=initial.catalog_generation_id,
        expected_generation_number=0,
        input_delta=input_delta("claimed_operation", (first,)),
        objects=(first,),
        reducer=reduce_complete_facts,
    )
    conflicting = fact("bundle", "run-b", 1)
    conflicting_delta = input_delta("claimed_operation", (conflicting,))

    with pytest.raises(CatalogOperationConflictError):
        store.rebase(
            input_delta=conflicting_delta,
            objects=(conflicting,),
            reducer=reduce_complete_facts,
        )

    conflicting_path = store.objects_path / f"{conflicting.fact_id}.json"
    assert not conflicting_path.exists()


def test_input_delta_requires_exact_objects_and_complete_dependencies(
    tmp_path: Path,
) -> None:
    store, initial = initialized_store(tmp_path / "catalog")
    record = fact("bundle", "run-a", 1)
    delta = input_delta("incomplete_operation", (record,))

    with pytest.raises(CatalogClosureError, match="exactly match"):
        store.publish(
            expected_generation_id=initial.catalog_generation_id,
            expected_generation_number=0,
            input_delta=delta,
            objects=(),
            reducer=reduce_complete_facts,
        )

    absent_dependency = content_id("test-catalog-fact", {"absent": True})
    forged = CatalogInputDelta.mint(
        scope_contract_id=delta.scope_contract_id,
        operation_id="missing_dependency_operation",
        configuration_fingerprint=delta.configuration_fingerprint,
        added_object_ids=delta.added_object_ids,
        dependency_closure_ids=tuple(sorted((record.fact_id, absent_dependency))),
    )
    with pytest.raises(CatalogClosureError, match="dependency closure"):
        store.publish(
            expected_generation_id=initial.catalog_generation_id,
            expected_generation_number=0,
            input_delta=forged,
            objects=(record,),
            reducer=reduce_complete_facts,
        )


def test_store_rejects_payloads_whose_attestation_is_excluded_from_identity(
    tmp_path: Path,
) -> None:
    store, initial = initialized_store(tmp_path / "catalog")
    attested = AttestedCatalogStoreFact.mint(
        value="scientific payload",
        attestation={"principal": "publisher-a"},
    )
    delta = CatalogInputDelta.mint(
        scope_contract_id=SCOPE_CONTRACT_ID,
        operation_id="attested_operation",
        configuration_fingerprint=CONFIGURATION_FINGERPRINT,
        added_object_ids=(attested.fact_id,),
        dependency_closure_ids=(attested.fact_id,),
    )

    with pytest.raises(ContractValidationError, match="separate immutable envelope"):
        store.publish(
            expected_generation_id=initial.catalog_generation_id,
            expected_generation_number=0,
            input_delta=delta,
            objects=(attested,),
            reducer=reduce_without_derived_references,
        )


def test_reducer_persists_target_states_separately_from_grow_only_source_facts(
    tmp_path: Path,
) -> None:
    store, initial = initialized_store(tmp_path / "catalog")
    first = fact("subject", "first", 1)
    first_commit = store.publish(
        expected_generation_id=initial.catalog_generation_id,
        expected_generation_number=0,
        input_delta=input_delta("derived_first", (first,)),
        objects=(first,),
        reducer=reduce_catalog_entry_states,
    )
    first_state_id = first_commit.generation.active_entry_state_ids[first.fact_id]
    first_state = store.read_contract(first_state_id, CatalogEntryState)

    second = fact("subject", "second", 1)
    second_delta = input_delta("derived_second", (second,))
    with pytest.raises(CatalogCompareAndSwapError):
        store.publish(
            expected_generation_id=initial.catalog_generation_id,
            expected_generation_number=0,
            input_delta=second_delta,
            objects=(second,),
            reducer=reduce_catalog_entry_states,
        )
    second_commit = store.rebase(
        input_delta=second_delta,
        objects=(second,),
        reducer=reduce_catalog_entry_states,
    )
    successor_state_id = second_commit.generation.active_entry_state_ids[first.fact_id]
    successor_state = store.read_contract(successor_state_id, CatalogEntryState)

    assert first_commit.generation.fact_object_ids == (first.fact_id,)
    assert first_commit.generation.derived_object_ids == (first_state_id,)
    assert second_commit.generation.fact_object_ids == tuple(
        sorted((first.fact_id, second.fact_id))
    )
    assert set(second_commit.generation.derived_object_ids) == set(
        second_commit.generation.active_entry_state_ids.values()
    )
    assert first_state.catalog_generation == 1
    assert successor_state.catalog_generation == 2
    assert successor_state.predecessor_state_id == first_state_id
    assert successor_state_id != first_state_id
    assert second_commit.delta_manifest is not None
    assert second_commit.delta_manifest.target_derived_object_ids == (
        second_commit.generation.derived_object_ids
    )


def test_catalog_entry_states_cannot_enter_as_precomputed_source_facts(
    tmp_path: Path,
) -> None:
    store, initial = initialized_store(tmp_path / "catalog")
    source = fact("subject", "source", 1)
    precomputed = CatalogEntryState.mint(
        subject_payload_id=source.fact_id,
        catalog_generation=1,
        predecessor_state_id=None,
        configuration_fingerprint=CONFIGURATION_FINGERPRINT,
        admission_state=AdmissionState.QUARANTINED,
        superseded_by_payload_ids=(),
        assertion_ids=(),
        revocation_ids=(),
        taint_source_ids=(),
    )
    delta = CatalogInputDelta.mint(
        scope_contract_id=SCOPE_CONTRACT_ID,
        operation_id="precomputed_state",
        configuration_fingerprint=CONFIGURATION_FINGERPRINT,
        added_object_ids=(precomputed.catalog_entry_state_id,),
        dependency_closure_ids=(precomputed.catalog_entry_state_id,),
    )

    with pytest.raises(CatalogClosureError, match="target reducer"):
        store.publish(
            expected_generation_id=initial.catalog_generation_id,
            expected_generation_number=0,
            input_delta=delta,
            objects=(precomputed,),
            reducer=reduce_catalog_entry_states,
        )


def test_reducer_state_for_wrong_generation_fails_before_derived_persistence(
    tmp_path: Path,
) -> None:
    store, initial = initialized_store(tmp_path / "catalog")
    source = fact("subject", "source", 1)

    with pytest.raises(CatalogReducerError, match="another generation"):
        store.publish(
            expected_generation_id=initial.catalog_generation_id,
            expected_generation_number=0,
            input_delta=input_delta("wrong_generation_state", (source,)),
            objects=(source,),
            reducer=reduce_states_for_previous_generation,
        )

    assert store.read_current() == initial


def test_crash_after_derived_state_persistence_reduces_the_same_target_on_retry(
    tmp_path: Path,
) -> None:
    store, initial = initialized_store(tmp_path / "catalog")
    source = fact("subject", "source", 1)
    delta = input_delta("derived_state_crash", (source,))

    def crash_after_derived_objects(event: str) -> None:
        if event == "derived_objects_persisted":
            raise InjectedCatalogCrash(event)

    with pytest.raises(InjectedCatalogCrash, match="derived_objects_persisted"):
        store.publish(
            expected_generation_id=initial.catalog_generation_id,
            expected_generation_number=0,
            input_delta=delta,
            objects=(source,),
            reducer=reduce_catalog_entry_states,
            crash_injector=crash_after_derived_objects,
        )
    assert store.read_current() == initial
    inactive_object_ids = {
        path.name.removesuffix(".json") for path in store.objects_path.iterdir()
    }

    recovered = store.rebase(
        input_delta=delta,
        objects=(source,),
        reducer=reduce_catalog_entry_states,
    )
    recovered_state_id = recovered.generation.active_entry_state_ids[source.fact_id]
    assert recovered_state_id in inactive_object_ids
    assert recovered.generation.derived_object_ids == (recovered_state_id,)


def test_reducer_cannot_remove_prior_derived_keys_before_pointer_publication(
    tmp_path: Path,
) -> None:
    store, initial = initialized_store(tmp_path / "catalog")
    first = fact("bundle", "run-a", 1)
    first_commit = store.publish(
        expected_generation_id=initial.catalog_generation_id,
        expected_generation_number=0,
        input_delta=input_delta("first_frontier", (first,)),
        objects=(first,),
        reducer=reduce_complete_facts,
    )
    second = fact("bundle", "run-b", 1)

    with pytest.raises(CatalogReducerError, match="removed a bundle frontier"):
        store.publish(
            expected_generation_id=first_commit.generation.catalog_generation_id,
            expected_generation_number=1,
            input_delta=input_delta("bad_reduction", (second,)),
            objects=(second,),
            reducer=reduce_without_derived_references,
        )

    assert store.read_current() == first_commit.generation


@pytest.mark.parametrize(
    ("crash_event", "expected_generation"),
    (
        ("objects_persisted", 0),
        ("input_delta_persisted", 0),
        ("derived_objects_persisted", 0),
        ("generation_persisted", 0),
        ("delta_manifest_persisted", 0),
        ("pointer_replaced", 1),
        ("pointer_directory_synced", 1),
    ),
)
def test_crash_boundaries_leave_old_or_complete_generation_and_retry_recovers(
    tmp_path: Path,
    crash_event: str,
    expected_generation: int,
) -> None:
    store, initial = initialized_store(tmp_path / crash_event)
    record = fact("bundle", crash_event, 1)
    delta = input_delta(f"operation_{crash_event}", (record,))

    def crash_at_event(event: str) -> None:
        if event == crash_event:
            raise InjectedCatalogCrash(event)

    with pytest.raises(InjectedCatalogCrash, match=crash_event):
        store.publish(
            expected_generation_id=initial.catalog_generation_id,
            expected_generation_number=0,
            input_delta=delta,
            objects=(record,),
            reducer=reduce_complete_facts,
            crash_injector=crash_at_event,
        )

    assert store.read_current().generation_number == expected_generation
    recovered = store.rebase(
        input_delta=delta,
        objects=(record,),
        reducer=reduce_complete_facts,
    )
    assert recovered.generation.generation_number == 1
    assert store.read_current() == recovered.generation


def test_initialization_recovers_an_inactive_durable_generation_zero(
    tmp_path: Path,
) -> None:
    store = CatalogStore(tmp_path / "catalog")

    def crash_after_generation(event: str) -> None:
        if event == "generation_persisted":
            raise InjectedCatalogCrash(event)

    with pytest.raises(InjectedCatalogCrash):
        store.initialize(
            scope_contract_id=SCOPE_CONTRACT_ID,
            configuration_fingerprint=CONFIGURATION_FINGERPRINT,
            crash_injector=crash_after_generation,
        )
    with pytest.raises(CatalogNotInitializedError):
        store.read_current()

    recovered = store.initialize(
        scope_contract_id=SCOPE_CONTRACT_ID,
        configuration_fingerprint=CONFIGURATION_FINGERPRINT,
    )
    assert recovered.generation_number == 0
    assert store.read_current() == recovered


def test_corrupt_immutable_object_is_detected_through_generation_closure(
    tmp_path: Path,
) -> None:
    store, initial = initialized_store(tmp_path / "catalog")
    record = fact("bundle", "run-a", 1)
    store.publish(
        expected_generation_id=initial.catalog_generation_id,
        expected_generation_number=0,
        input_delta=input_delta("corruption_operation", (record,)),
        objects=(record,),
        reducer=reduce_complete_facts,
    )
    object_path = store.objects_path / f"{record.fact_id}.json"
    object_path.write_bytes(b'{"invalid":"bytes"}')

    with pytest.raises(CatalogCorruptionError):
        store.read_current()


def test_symlinked_layout_and_pointer_are_rejected(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    symlink_root = tmp_path / "catalog-link"
    symlink_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(CatalogLayoutError, match="symlink"):
        CatalogStore(symlink_root)

    store, _ = initialized_store(tmp_path / "catalog")
    store.current_path.unlink()
    store.current_path.symlink_to(tmp_path / "missing-pointer")
    with pytest.raises(CatalogLayoutError, match="regular file"):
        store.read_current()


def test_two_os_process_rebases_serialize_into_one_grow_only_lineage(
    tmp_path: Path,
) -> None:
    root = tmp_path / "catalog"
    store, _ = initialized_store(root)
    first = fact("bundle", "run-a", 1)
    second = fact("bundle", "run-b", 1)
    first_delta = input_delta("process_a", (first,))
    second_delta = input_delta("process_b", (second,))
    context = multiprocessing.get_context("fork")
    start = context.Event()
    completed = context.Queue()
    processes = (
        context.Process(
            target=publish_rebased_in_process,
            args=(str(root), first_delta, first, start, completed),
        ),
        context.Process(
            target=publish_rebased_in_process,
            args=(str(root), second_delta, second, start, completed),
        ),
    )
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join()

    assert tuple(process.exitcode for process in processes) == (0, 0)
    assert len({completed.get(), completed.get()}) == 2
    current = store.read_current()
    assert current.generation_number == 2
    assert current.fact_object_ids == tuple(sorted((first.fact_id, second.fact_id)))
    assert dict(current.bundle_frontier) == {
        "run-a": first.fact_id,
        "run-b": second.fact_id,
    }


def test_two_os_processes_racing_one_expected_pointer_allow_one_cas_winner(
    tmp_path: Path,
) -> None:
    root = tmp_path / "catalog"
    store, initial = initialized_store(root)
    first = fact("bundle", "run-a", 1)
    second = fact("bundle", "run-b", 1)
    first_delta = input_delta("cas_process_a", (first,))
    second_delta = input_delta("cas_process_b", (second,))
    context = multiprocessing.get_context("fork")
    start = context.Event()
    completed = context.Queue()
    processes = (
        context.Process(
            target=publish_expected_in_process,
            args=(str(root), initial, first_delta, first, start, completed),
        ),
        context.Process(
            target=publish_expected_in_process,
            args=(str(root), initial, second_delta, second, start, completed),
        ),
    )
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join()

    assert sorted(process.exitcode for process in processes) == [0, 1]
    winning_generation_id = completed.get()
    current = store.read_current()
    assert current.catalog_generation_id == winning_generation_id
    assert current.generation_number == 1
    assert len(current.fact_object_ids) == 1
