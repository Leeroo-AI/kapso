"""Single-lock publication of checkpoint-governed run-state generations."""

from __future__ import annotations

import os
import stat
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
from threading import Barrier

import pytest

from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.launch.checkpoint_contracts import (
    RunCheckpoint,
    RunCheckpointHead,
    RunCheckpointStatus,
    RunCheckpointStop,
)
from kapso.cross_run.launch.checkpoint_control import RunCheckpointControlError
from kapso.cross_run.launch.derived_state_contracts import (
    RunStateAuthority,
    RunStateLayout,
)
from kapso.cross_run.launch.resume_contracts import (
    RunDerivativeFrontier,
    RunReleaseUseMode,
    RunSafetyBoundary,
    RunSafetyState,
)
from kapso.cross_run.launch.run_state_publisher import (
    ReconciledRunFrontier,
    RunStatePublisher,
    RunStatePublisherError,
)
from kapso.cross_run.launch.workspace import (
    LaunchWorkspaceError,
    StarterWorkspaceBuilder,
)
from test_launch_checkpoint_contracts import _successor_safety
from test_launch_resolver import resolver_case
from test_launch_resume_contracts import (
    _empty_frontier,
    _remint_evidence,
    _security_observation,
    _subjects,
)
from test_reconciled_run_state_projection import _resolved_projection


def _layout(active, strategy_kind="generic") -> RunStateLayout:
    installed = active.bootstrap_pin.installation_receipt.layout
    authority_paths = {
        RunStateAuthority.EXPERIMENT_HISTORY: (
            installed.run_experiment_history_relative_path
        ),
        RunStateAuthority.EXECUTION_JOURNAL: (
            installed.run_execution_journal_relative_path
        ),
    }
    if strategy_kind == "generic":
        authority_paths[RunStateAuthority.IDEA_ARCHIVE] = (
            installed.run_idea_archive_relative_path
        )
    return RunStateLayout.build(
        strategy_kind=strategy_kind,
        authority_paths=authority_paths,
    )


def _initial_safety(
    active,
    projection,
    *,
    security_generation_offset=1,
) -> RunSafetyState:
    pin = active.bootstrap_pin
    empty_frontier = _empty_frontier(pin)
    payloads = projection.payload_by_authority
    evidence = _remint_evidence(
        empty_frontier.evidence,
        state_authority_digests={
            authority.value: tree_or_blob_digest(payload)
            for authority, payload in payloads.items()
        },
        state_authority_revisions={
            authority.value: revision
            for authority, revision in projection.revision_by_authority.items()
        },
    )
    frontier = RunDerivativeFrontier.build(
        launch_subject_ids=empty_frontier.launch_subject_ids,
        evidence=evidence,
        derivatives=(),
    )
    release_use = pin.launch_manifest.release_use_observation
    return RunSafetyState.build(
        predecessor=None,
        bootstrap_pin=pin,
        boundary=RunSafetyBoundary.INITIALIZATION,
        derivative_frontier=frontier,
        security_observation=_security_observation(
            pin,
            _subjects(pin, release_use, frontier),
            generation_offset=security_generation_offset,
        ),
        release_use_observation=release_use,
        release_use_mode=RunReleaseUseMode.ONLINE_CURRENT,
    )


def _genesis(
    active,
    resolver_case,
    *,
    security_generation_offset=1,
    strategy_kind="generic",
):
    pin = active.bootstrap_pin
    projection = _resolved_projection(strategy_kind, pin, resolver_case)
    safety = _initial_safety(
        active,
        projection,
        security_generation_offset=security_generation_offset,
    )
    bundle = projection.build_bundle(
        bootstrap_pin=pin,
        run_state_layout=_layout(active, strategy_kind),
        predecessor_checkpoint_head_id=(
            RunCheckpointHead.initial(pin).run_checkpoint_head_id
        ),
        predecessor_checkpoint_id=None,
        predecessor_evidence_id=None,
        target_evidence_id=safety.derivative_frontier.evidence.evidence_id,
        predecessor_bundle=None,
        predecessor_strategy_state=None,
    )
    checkpoint = RunCheckpoint.build(
        predecessor=None,
        status=RunCheckpointStatus.ACTIVE,
        last_stop=None,
        completed_iterations=0,
        cumulative_cost=0.0,
        elapsed_seconds=0.0,
        cost_by_component={},
        feedback_source=None,
        current_feedback=None,
        termination_decision=None,
        strategy_state=projection.strategy_state,
        safety_state=safety,
        derived_state_generation=bundle.generation,
    )
    return projection, bundle, checkpoint


def _successor(active, projection, predecessor_bundle, predecessor_checkpoint):
    pin = active.bootstrap_pin
    safety = _successor_safety(pin, predecessor_checkpoint.safety_state)
    predecessor_head = RunCheckpointHead.initial(pin).advance(predecessor_checkpoint)
    bundle = projection.build_bundle(
        bootstrap_pin=pin,
        run_state_layout=_layout(active),
        predecessor_checkpoint_head_id=predecessor_head.run_checkpoint_head_id,
        predecessor_checkpoint_id=predecessor_checkpoint.run_checkpoint_id,
        predecessor_evidence_id=(
            predecessor_checkpoint.safety_state.derivative_frontier.evidence.evidence_id
        ),
        target_evidence_id=safety.derivative_frontier.evidence.evidence_id,
        predecessor_bundle=predecessor_bundle,
        predecessor_strategy_state=predecessor_checkpoint.strategy_state,
    )
    checkpoint = RunCheckpoint.build(
        predecessor=predecessor_checkpoint,
        status=RunCheckpointStatus.ACTIVE,
        last_stop=RunCheckpointStop.COST_BUDGET,
        completed_iterations=0,
        cumulative_cost=1.0,
        elapsed_seconds=1.0,
        cost_by_component={"implementation": 1.0},
        feedback_source=None,
        current_feedback=None,
        termination_decision=None,
        strategy_state=projection.strategy_state,
        safety_state=safety,
        derived_state_generation=bundle.generation,
    )
    return bundle, checkpoint


@pytest.fixture
def publisher_case(resolver_case, tmp_path):
    resolved = resolver_case["resolver"].resolve(resolver_case["request"])
    prepared = StarterWorkspaceBuilder(resolver_case["resolver"]._settings).build(
        resolved,
        (tmp_path / "run").absolute(),
        run_id="run-state-publisher",
        campaign_id="campaign-state-publisher",
    )
    active = prepared.require_builder_authority()
    projection, bundle, checkpoint = _genesis(active, resolver_case)
    return {
        "active": active,
        "settings": resolver_case["resolver"]._settings.launch,
        "projection": projection,
        "bundle": bundle,
        "checkpoint": checkpoint,
    }


def _publish_genesis(case):
    publisher = RunStatePublisher(case["active"], case["settings"])
    permit = publisher.issue_publication_permit(
        None,
        case["checkpoint"],
        case["bundle"],
    )
    receipt = publisher.publish(
        permit,
        case["checkpoint"],
        case["bundle"],
    )
    return publisher, receipt


def _generation_path(case):
    layout = case["active"].bootstrap_pin.installation_receipt.layout
    return (
        case["active"].run_root
        / layout.run_derived_state_store_relative_path
        / case["bundle"].object_name
    )


def _generation_store_path(case):
    layout = case["active"].bootstrap_pin.installation_receipt.layout
    return case["active"].run_root / layout.run_derived_state_store_relative_path


def _write_durable_file(path, payload: bytes, mode: int) -> None:
    path.write_bytes(payload)
    path.chmod(mode)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    os.fsync(descriptor)
    os.close(descriptor)
    parent_descriptor = os.open(
        path.parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    os.fsync(parent_descriptor)
    os.close(parent_descriptor)


def _append_durable_file(path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with os.fdopen(descriptor, "ab", buffering=0) as handle:
        written = handle.write(payload)
        if written != len(payload):
            raise AssertionError("test journal append was incomplete")
        os.fsync(handle.fileno())


def _write_checkpoint_ahead(case) -> None:
    checkpoint_path = case["active"].run_root / case["settings"].run_checkpoint_path
    _write_durable_file(
        checkpoint_path,
        case["checkpoint"].to_json_bytes(),
        0o400,
    )


def test_concurrent_distinct_genesis_permits_allow_exactly_one_publication(
    publisher_case,
    resolver_case,
) -> None:
    publisher = RunStatePublisher(
        publisher_case["active"],
        publisher_case["settings"],
    )
    _alternate_projection, alternate_bundle, alternate_checkpoint = _genesis(
        publisher_case["active"],
        resolver_case,
        security_generation_offset=2,
    )
    assert alternate_checkpoint != publisher_case["checkpoint"]
    permits_and_candidates = (
        (
            publisher.issue_publication_permit(
                None,
                publisher_case["checkpoint"],
                publisher_case["bundle"],
            ),
            publisher_case["checkpoint"],
            publisher_case["bundle"],
        ),
        (
            publisher.issue_publication_permit(
                None,
                alternate_checkpoint,
                alternate_bundle,
            ),
            alternate_checkpoint,
            alternate_bundle,
        ),
    )
    barrier = Barrier(len(permits_and_candidates))

    def publish_after_barrier(permit, checkpoint, bundle):
        barrier.wait()
        return publisher.publish(permit, checkpoint, bundle)

    with ThreadPoolExecutor(max_workers=len(permits_and_candidates)) as executor:
        futures = tuple(
            executor.submit(
                publish_after_barrier,
                permit,
                checkpoint,
                bundle,
            )
            for permit, checkpoint, bundle in permits_and_candidates
        )
    errors = tuple(future.exception() for future in futures)
    receipts = tuple(
        future.result()
        for future, error in zip(futures, errors, strict=True)
        if error is None
    )
    rejected = tuple(error for error in errors if error is not None)

    assert len(receipts) == 1
    assert len(rejected) == 1
    assert isinstance(rejected[0], RunStatePublisherError)
    assert "frontier moved" in str(rejected[0])
    assert receipts[0].checkpoint in (
        publisher_case["checkpoint"],
        alternate_checkpoint,
    )
    current = publisher.load_reconciled()
    assert current is not None
    assert current.checkpoint == receipts[0].checkpoint


def test_fresh_idempotent_retry_reuses_one_generation_object(
    publisher_case,
) -> None:
    publisher, first = _publish_genesis(publisher_case)
    store = _generation_store_path(publisher_case)
    first_entries = tuple(sorted(path.name for path in store.iterdir()))
    first_identity = (
        _generation_path(publisher_case).stat().st_dev,
        _generation_path(publisher_case).stat().st_ino,
    )

    permit = publisher.issue_publication_permit(
        first,
        publisher_case["checkpoint"],
        publisher_case["bundle"],
    )
    retried = publisher.publish(
        permit,
        publisher_case["checkpoint"],
        publisher_case["bundle"],
    )

    assert retried.checkpoint == first.checkpoint
    assert retried.bundle == first.bundle
    assert tuple(sorted(path.name for path in store.iterdir())) == first_entries
    assert len(first_entries) == 1
    assert (
        _generation_path(publisher_case).stat().st_dev,
        _generation_path(publisher_case).stat().st_ino,
    ) == first_identity


def test_failure_after_bundle_publication_leaves_reusable_orphan(
    publisher_case,
    monkeypatch,
) -> None:
    publisher = RunStatePublisher(
        publisher_case["active"],
        publisher_case["settings"],
    )
    permit = publisher.issue_publication_permit(
        None,
        publisher_case["checkpoint"],
        publisher_case["bundle"],
    )

    def fail_before_checkpoint(*_args, **_kwargs):
        raise RuntimeError("injected failure before checkpoint")

    monkeypatch.setattr(
        publisher._checkpoint,
        "_commit_checkpoint",
        fail_before_checkpoint,
    )
    with pytest.raises(RuntimeError, match="injected failure"):
        publisher.publish(
            permit,
            publisher_case["checkpoint"],
            publisher_case["bundle"],
        )

    orphan = _generation_path(publisher_case)
    orphan_identity = (orphan.stat().st_dev, orphan.stat().st_ino)
    checkpoint_path = (
        publisher_case["active"].run_root
        / publisher_case["settings"].run_checkpoint_path
    )
    assert orphan.read_bytes() == publisher_case["bundle"].to_bytes()
    assert not checkpoint_path.exists()

    recovered = RunStatePublisher(
        publisher_case["active"],
        publisher_case["settings"],
    )
    recovery_permit = recovered.issue_publication_permit(
        None,
        publisher_case["checkpoint"],
        publisher_case["bundle"],
    )
    receipt = recovered.publish(
        recovery_permit,
        publisher_case["checkpoint"],
        publisher_case["bundle"],
    )

    assert receipt.require_current(recovered) == publisher_case["checkpoint"]
    assert (orphan.stat().st_dev, orphan.stat().st_ino) == orphan_identity
    assert len(tuple(_generation_store_path(publisher_case).iterdir())) == 1


@pytest.mark.parametrize("tail_kind", ("exact_prefix", "unrelated"))
def test_checkpoint_journal_recovery_accepts_only_exact_record_prefix(
    publisher_case,
    tail_kind,
) -> None:
    publisher = RunStatePublisher(
        publisher_case["active"],
        publisher_case["settings"],
    )
    _write_durable_file(
        _generation_path(publisher_case),
        publisher_case["bundle"].to_bytes(),
        0o400,
    )
    _write_checkpoint_ahead(publisher_case)
    journal_path = (
        publisher_case["active"].run_root
        / publisher_case["settings"].run_checkpoint_journal_path
    )
    initial_journal = journal_path.read_bytes()
    successor_record = (
        RunCheckpointHead.initial(publisher_case["active"].bootstrap_pin)
        .advance(publisher_case["checkpoint"])
        .to_json_bytes()
        + b"\n"
    )
    tail = (
        successor_record[: len(successor_record) // 2]
        if tail_kind == "exact_prefix"
        else b"!"
    )
    _append_durable_file(journal_path, tail)

    if tail_kind == "exact_prefix":
        receipt = publisher.load_reconciled()
        assert receipt.require_current(publisher) == publisher_case["checkpoint"]
        assert journal_path.read_bytes() == initial_journal + successor_record
    else:
        with pytest.raises(
            RunCheckpointControlError,
            match="exact recovery record prefix",
        ):
            publisher.load_reconciled()
        assert journal_path.read_bytes() == initial_journal + tail


def test_store_capacity_rejects_absent_target_but_accepts_existing_target(
    publisher_case,
) -> None:
    publisher = RunStatePublisher(
        publisher_case["active"],
        publisher_case["settings"],
    )
    publisher._settings = replace(
        publisher._settings,
        run_derived_state_store_entry_limit=1,
    )
    store = _generation_store_path(publisher_case)
    target = _generation_path(publisher_case)
    occupied_name = (
        f"generation-{'0' * 64}.bundle"
        if target.name != f"generation-{'0' * 64}.bundle"
        else f"generation-{'1' * 64}.bundle"
    )
    occupied = store / occupied_name
    _write_durable_file(occupied, b"occupied", 0o400)
    journal_path = (
        publisher_case["active"].run_root
        / publisher_case["settings"].run_checkpoint_journal_path
    )
    initial_journal = journal_path.read_bytes()
    checkpoint_path = (
        publisher_case["active"].run_root
        / publisher_case["settings"].run_checkpoint_path
    )
    permit = publisher.issue_publication_permit(
        None,
        publisher_case["checkpoint"],
        publisher_case["bundle"],
    )

    with pytest.raises(RunStatePublisherError, match="no publication capacity"):
        publisher.publish(
            permit,
            publisher_case["checkpoint"],
            publisher_case["bundle"],
        )

    assert not checkpoint_path.exists()
    assert journal_path.read_bytes() == initial_journal
    assert not target.exists()

    occupied.unlink()
    _write_durable_file(
        target,
        publisher_case["bundle"].to_bytes(),
        0o400,
    )
    retry_permit = publisher.issue_publication_permit(
        None,
        publisher_case["checkpoint"],
        publisher_case["bundle"],
    )
    receipt = publisher.publish(
        retry_permit,
        publisher_case["checkpoint"],
        publisher_case["bundle"],
    )

    assert receipt.require_current(publisher) == publisher_case["checkpoint"]
    assert tuple(path.name for path in store.iterdir()) == (target.name,)


def test_publisher_publishes_reconciled_genesis_and_exact_views(
    publisher_case,
) -> None:
    publisher, receipt = _publish_genesis(publisher_case)
    bundle = publisher_case["bundle"]

    assert type(receipt) is ReconciledRunFrontier
    assert receipt.require_current(publisher) == publisher_case["checkpoint"]
    assert _generation_path(publisher_case).read_bytes() == bundle.to_bytes()
    assert stat.S_IMODE(_generation_path(publisher_case).stat().st_mode) == 0o400
    for relative_path, payload in bundle.payload_by_relative_path().items():
        view = publisher_case["active"].run_root / relative_path
        assert view.read_bytes() == payload
        assert stat.S_IMODE(view.stat().st_mode) == 0o400
        assert view.stat().st_nlink == 1
    loaded = publisher.load_reconciled()
    assert type(loaded) is ReconciledRunFrontier
    assert loaded.require_current(publisher) == publisher_case["checkpoint"]


def test_tree_publication_owns_only_history_and_journal(
    resolver_case,
    tmp_path,
) -> None:
    request = resolver_case["request"]
    request_values = {
        field.name: getattr(request, field.name)
        for field in fields(request)
        if field.name != "launch_request_id"
    }
    request_values["search_mode"] = "benchmark_tree_search"
    resolved = resolver_case["resolver"].resolve(type(request).mint(**request_values))
    prepared = StarterWorkspaceBuilder(resolver_case["resolver"]._settings).build(
        resolved,
        (tmp_path / "tree-run").absolute(),
        run_id="run-state-publisher-tree",
        campaign_id="campaign-state-publisher-tree",
    )
    active = prepared.require_builder_authority()
    settings = resolver_case["resolver"]._settings.launch
    projection, bundle, checkpoint = _genesis(
        active,
        resolver_case,
        strategy_kind="benchmark_tree_search",
    )
    publisher = RunStatePublisher(active, settings)
    receipt = publisher.publish(
        publisher.issue_publication_permit(None, checkpoint, bundle),
        checkpoint,
        bundle,
    )
    idea_path = active.run_root / settings.run_idea_archive_path

    assert receipt.projection == projection
    assert receipt.require_current(publisher) == checkpoint
    assert not idea_path.exists()
    assert set(bundle.payload_by_relative_path()) == {
        settings.run_experiment_history_path,
        settings.run_execution_journal_path,
    }

    _write_durable_file(idea_path, b"{}", 0o400)
    with pytest.raises(RunStatePublisherError, match="stray idea archive"):
        publisher.load_reconciled()


def test_publication_permit_is_nonclonable_and_one_shot(publisher_case) -> None:
    publisher = RunStatePublisher(
        publisher_case["active"],
        publisher_case["settings"],
    )
    permit = publisher.issue_publication_permit(
        None,
        publisher_case["checkpoint"],
        publisher_case["bundle"],
    )

    with pytest.raises(
        RunStatePublisherError,
        match="cloned|foreign|consumed",
    ):
        publisher.publish(
            replace(permit),
            publisher_case["checkpoint"],
            publisher_case["bundle"],
        )

    publisher.publish(
        permit,
        publisher_case["checkpoint"],
        publisher_case["bundle"],
    )
    with pytest.raises(
        RunStatePublisherError,
        match="cloned|foreign|consumed",
    ):
        publisher.publish(
            permit,
            publisher_case["checkpoint"],
            publisher_case["bundle"],
        )


def test_reconciled_receipt_is_nonclonable_and_stales_after_successor(
    publisher_case,
) -> None:
    publisher, receipt = _publish_genesis(publisher_case)

    with pytest.raises(RunStatePublisherError, match="cloned|foreign"):
        replace(receipt).require_current(publisher)

    successor_bundle, successor_checkpoint = _successor(
        publisher_case["active"],
        publisher_case["projection"],
        publisher_case["bundle"],
        publisher_case["checkpoint"],
    )
    permit = publisher.issue_publication_permit(
        receipt,
        successor_checkpoint,
        successor_bundle,
    )
    successor_receipt = publisher.publish(
        permit,
        successor_checkpoint,
        successor_bundle,
    )

    assert successor_receipt.require_current(publisher) == successor_checkpoint
    with pytest.raises(
        RunStatePublisherError,
        match="no longer current|stale",
    ):
        receipt.require_current(publisher)


@pytest.mark.parametrize("frontier_kind", ("checkpoint_id", "clone"))
def test_successor_publication_requires_live_reconciled_frontier(
    publisher_case,
    frontier_kind,
) -> None:
    publisher, receipt = _publish_genesis(publisher_case)
    successor_bundle, successor_checkpoint = _successor(
        publisher_case["active"],
        publisher_case["projection"],
        publisher_case["bundle"],
        publisher_case["checkpoint"],
    )
    observed_frontier = (
        receipt.run_checkpoint_id
        if frontier_kind == "checkpoint_id"
        else replace(receipt)
    )

    with pytest.raises(
        RunStatePublisherError,
        match="exact receipt|cloned|foreign",
    ):
        publisher.issue_publication_permit(
            observed_frontier,
            successor_checkpoint,
            successor_bundle,
        )

    assert publisher.require_current(receipt) == publisher_case["checkpoint"]


def test_load_reconciled_recovers_checkpoint_ahead_when_bundle_is_durable(
    publisher_case,
) -> None:
    publisher = RunStatePublisher(
        publisher_case["active"],
        publisher_case["settings"],
    )
    _write_durable_file(
        _generation_path(publisher_case),
        publisher_case["bundle"].to_bytes(),
        0o400,
    )
    _write_checkpoint_ahead(publisher_case)

    receipt = publisher.load_reconciled()

    assert receipt.require_current(publisher) == publisher_case["checkpoint"]
    for relative_path, payload in (
        publisher_case["bundle"].payload_by_relative_path().items()
    ):
        assert (publisher_case["active"].run_root / relative_path).read_bytes() == (
            payload
        )


def test_checkpoint_ahead_without_referenced_bundle_fails_closed(
    publisher_case,
) -> None:
    _write_checkpoint_ahead(publisher_case)
    journal_path = (
        publisher_case["active"].run_root
        / publisher_case["settings"].run_checkpoint_journal_path
    )
    initial_journal = journal_path.read_bytes()
    publisher = RunStatePublisher(
        publisher_case["active"],
        publisher_case["settings"],
    )

    assert journal_path.read_bytes() == initial_journal

    with pytest.raises(
        RunStatePublisherError,
        match="bundle|generation|absent",
    ):
        publisher.load_reconciled()


def test_load_reconciled_repairs_missing_and_stale_views(publisher_case) -> None:
    publisher, _receipt = _publish_genesis(publisher_case)
    views = tuple(
        (
            publisher_case["active"].run_root / relative_path,
            payload,
        )
        for relative_path, payload in publisher_case["bundle"]
        .payload_by_relative_path()
        .items()
    )
    views[0][0].unlink()
    views[1][0].chmod(0o600)
    views[1][0].write_bytes(b'{"stale":"repairable"}')
    views[1][0].chmod(0o400)

    receipt = publisher.load_reconciled()

    assert receipt.require_current(publisher) == publisher_case["checkpoint"]
    for view, payload in views:
        assert view.read_bytes() == payload
        assert stat.S_IMODE(view.stat().st_mode) == 0o400


def test_load_reconciled_rejects_corrupt_referenced_bundle(
    publisher_case,
) -> None:
    publisher, _receipt = _publish_genesis(publisher_case)
    generation = _generation_path(publisher_case)
    generation.chmod(0o600)
    generation.write_bytes(b"corrupt retained generation")
    generation.chmod(0o400)

    with pytest.raises(
        (RunStatePublisherError, ValueError),
        match="bundle|generation|canonical|header|digest",
    ):
        publisher.load_reconciled()


def test_load_reconciled_rejects_unsafe_view(publisher_case) -> None:
    publisher, _receipt = _publish_genesis(publisher_case)
    relative_path = next(iter(publisher_case["bundle"].payload_by_relative_path()))
    view = publisher_case["active"].run_root / relative_path
    view.chmod(0o600)

    with pytest.raises(
        (RunStatePublisherError, LaunchWorkspaceError),
        match="unsafe|projection|owner-private",
    ):
        publisher.load_reconciled()
