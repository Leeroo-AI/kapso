import os
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest

from kapso.cross_run.expert import (
    ExpertCandidateStore,
    ExpertCandidateStoreError,
    ExpertCandidateValidator,
)
from test_expert_candidates import (
    bootstrap_candidate_closure,
    expert_settings,
    sanitation_settings,
)


def candidate_store(tmp_path):
    tmp_path.chmod(0o700)
    return ExpertCandidateStore(
        tmp_path / "candidates",
        tmp_path,
        ExpertCandidateValidator(expert_settings(), sanitation_settings()),
    )


class SplitViewMapping(Mapping):
    def __init__(self, contents):
        self.contents = contents

    def __iter__(self):
        return iter(self.contents)

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, key):
        return self.contents[key]

    def items(self):
        return ((path, b"mutated second view") for path in self.contents)


def test_candidate_store_seals_and_reopens_exact_closure(tmp_path):
    store = candidate_store(tmp_path)
    closure = bootstrap_candidate_closure()

    stored = store.persist(closure)
    reopened = store.read(closure.manifest.candidate_id)
    replayed = store.persist(closure)

    assert reopened == stored
    assert replayed == stored
    assert reopened.closure == closure
    assert reopened.root.parent == store.object_root
    assert (reopened.root / "COMMITTED.json").is_file()
    assert (
        reopened.root / "derivations/agent/workspace-delta.json"
    ).read_bytes() == closure.derivation.workspace_delta.to_json_bytes()
    assert (
        reopened.root / "validation-context.json"
    ).read_bytes() == closure.validation_context.to_json_bytes()
    assert tuple(store.staging_root.iterdir()) == ()


def test_candidate_store_serializes_concurrent_identical_replay(tmp_path):
    store = candidate_store(tmp_path)
    closure = bootstrap_candidate_closure()

    with ThreadPoolExecutor(max_workers=2) as executor:
        stored = tuple(executor.map(store.persist, (closure, closure)))

    assert stored[0] == stored[1]
    assert tuple(path.name for path in store.object_root.iterdir()) == (
        closure.manifest.candidate_id.rsplit(":", 1)[1],
    )


def test_candidate_store_serializes_concurrent_first_construction(tmp_path):
    tmp_path.chmod(0o700)

    with ThreadPoolExecutor(max_workers=2) as executor:
        stores = tuple(executor.map(candidate_store, (tmp_path, tmp_path)))

    assert stores[0].root == stores[1].root
    assert stores[0].object_root.is_dir()
    assert stores[0].staging_root.is_dir()


def test_candidate_store_packages_one_validated_byte_snapshot(tmp_path):
    store = candidate_store(tmp_path)
    closure = bootstrap_candidate_closure()
    changing_view = replace(
        closure,
        candidate_contents=SplitViewMapping(closure.candidate_contents),
    )

    stored = store.persist(changing_view)

    assert stored.closure.candidate_contents == closure.candidate_contents


def test_candidate_store_recovers_uncommitted_staging(tmp_path):
    store = candidate_store(tmp_path)
    stale = store.staging_root / ".candidate-stale"
    stale.mkdir(mode=0o700)
    stale_file = stale / "partial.json"
    stale_file.write_bytes(b"partial")
    stale_file.chmod(0o600)

    reopened = candidate_store(tmp_path)

    assert reopened.root == store.root
    assert not stale.exists()


def test_candidate_store_rejects_checksum_and_hardlink_corruption(tmp_path):
    store = candidate_store(tmp_path)
    closure = bootstrap_candidate_closure()
    stored = store.persist(closure)
    source = stored.root / "source/src/execution.py"
    source.write_bytes(b"changed after commit")

    with pytest.raises(ExpertCandidateStoreError, match="checksum differs"):
        store.read(closure.manifest.candidate_id)

    source.write_bytes(closure.candidate_contents["src/execution.py"])
    hardlink = tmp_path / "linked-source"
    os.link(source, hardlink)
    with pytest.raises(ExpertCandidateStoreError, match="independent file"):
        store.read(closure.manifest.candidate_id)


def test_candidate_store_rejects_noncanonical_commit_record(tmp_path):
    store = candidate_store(tmp_path)
    closure = bootstrap_candidate_closure()
    stored = store.persist(closure)
    commit_path = stored.root / "COMMITTED.json"
    commit_path.write_bytes(commit_path.read_bytes() + b"\n")

    with pytest.raises(ExpertCandidateStoreError, match="not canonical"):
        store.read(closure.manifest.candidate_id)


def test_candidate_store_rejects_agent_artifact_corruption(tmp_path):
    store = candidate_store(tmp_path)
    closure = bootstrap_candidate_closure()
    stored = store.persist(closure)
    prompt = stored.root / "derivations/agent/artifacts/prompt.txt"
    prompt.write_bytes(b"substituted prompt")

    with pytest.raises(ExpertCandidateStoreError, match="checksum differs"):
        store.read(closure.manifest.candidate_id)


def test_candidate_store_syncs_both_rename_parents(tmp_path, monkeypatch):
    synchronized: list[Path] = []
    synchronize = ExpertCandidateStore._fsync_directory

    def record_synchronization(path):
        synchronized.append(path)
        synchronize(path)

    monkeypatch.setattr(
        ExpertCandidateStore,
        "_fsync_directory",
        staticmethod(record_synchronization),
    )
    store = candidate_store(tmp_path)

    store.persist(bootstrap_candidate_closure())

    assert store.staging_root in synchronized
    assert store.object_root in synchronized


@pytest.mark.parametrize("corruption", ("symlink", "fifo", "public_mode"))
def test_candidate_store_rejects_unsafe_package_entries(tmp_path, corruption):
    store = candidate_store(tmp_path)
    closure = bootstrap_candidate_closure()
    stored = store.persist(closure)
    source = stored.root / "source/src/execution.py"
    if corruption == "symlink":
        source.unlink()
        source.symlink_to(tmp_path / "outside")
    elif corruption == "fifo":
        source.unlink()
        os.mkfifo(source, mode=0o600)
    else:
        source.chmod(0o644)

    with pytest.raises(ExpertCandidateStoreError, match="candidate package"):
        store.read(closure.manifest.candidate_id)
