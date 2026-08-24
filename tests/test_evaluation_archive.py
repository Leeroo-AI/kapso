"""The evaluation-archive contract: stamps, snapshots, head, final selection.

This is the machinery that keeps the shipped pick in sync with the in-loop
evaluation when the evaluator changes mid-run (EVALUATION_GOVERNANCE.md).
Every test pins either a fail-loud path or a doctrine invariant: measurements
never cross evaluator boundaries, missing evidence never wins, and stored
scores are only ever a tamper tripwire against their own recomputation.
"""

from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path

import pytest

from kapso.execution import evaluation_archive as archive
from kapso.execution import evaluation_archive_sandbox as sandbox
from kapso.execution.evaluation_integrity import (
    build_evaluation_manifest,
    manifest_fingerprint,
)

PROVIDED_GRADER_BYTES = b"# provided grader: immutable scoring core\n"

FAKE_ENTRYPOINT = textwrap.dedent(
    """
    import argparse
    import json
    from pathlib import Path

    MARKER = "KAPSO_EVAL_MANIFEST"

    parser = argparse.ArgumentParser()
    parser.add_argument("--rescore")
    parser.add_argument("--fidelity")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    run = Path(args.rescore)
    truth = json.loads((run / "truth.json").read_text())
    label = json.loads((run / "private" / "selection.json").read_text())
    line = {
        "fidelity": "full",
        "fraction": 1.0,
        "seed": 1337,
        "items": truth["items"],
        "total_items": truth["items"],
        "score": truth["score"],
        "run": run.name,
        "session": label["session"],
        "mode": "rescore",
        "metrics": truth.get("metrics", {}),
    }
    print(MARKER + " " + json.dumps(line))
    """
)


def make_evaluation_source(root: Path) -> Path:
    """A minimal registered tree: maintainer entrypoint + provided grader."""
    source = root / "kapso_evaluation"
    source.mkdir()
    (source / "kapso_eval.py").write_text(FAKE_ENTRYPOINT)
    (source / "grader.py").write_bytes(PROVIDED_GRADER_BYTES)
    return source


def make_run(
    runs_root: Path,
    name: str,
    *,
    session: str,
    score: float,
    evaluator_id: str,
    status: str = "pending",
    metrics: dict | None = None,
) -> Path:
    """One archived run exactly as the sandbox helper would record it.

    truth.json stands in for the stored predictions the entrypoint rescored
    from; manifest.txt is the archive-time score of record the tripwire
    checks against.
    """
    run_dir = runs_root / name
    (run_dir / "private").mkdir(parents=True)
    (run_dir / "truth.json").write_text(
        json.dumps({"items": 10, "score": score, "metrics": metrics or {}})
    )
    line = {
        "fidelity": "full",
        "fraction": 1.0,
        "seed": 1337,
        "items": 10,
        "total_items": 10,
        "score": score,
        "run": name,
        "session": session,
    }
    (run_dir / "manifest.txt").write_text(
        sandbox.MANIFEST_MARKER + " " + json.dumps(line) + "\n"
    )
    record = {
        "status": status,
        "session": session,
        "by": "grader",
        "evaluator_id": evaluator_id,
    }
    (run_dir / "private" / "selection.json").write_text(json.dumps(record))
    return run_dir


@pytest.fixture()
def governed_archive(tmp_path):
    """An archive with a snapshotted evaluator, ready for runs."""
    source = make_evaluation_source(tmp_path)
    head = sandbox.fingerprint_tree(source)
    archive_root = tmp_path / "work"
    (archive_root / "runs").mkdir(parents=True)
    sandbox.snapshot_evaluator_tree(archive_root, source, head)
    return archive_root, head, source


class TestSandboxFingerprintMirrorsFramework:
    """The sandbox stamp must equal the registered fingerprint, or the whole
    pooling scheme silently partitions on the wrong key."""

    def test_clean_tree_agrees_with_framework_fingerprint(self, tmp_path):
        source = make_evaluation_source(tmp_path)
        assert sandbox.fingerprint_tree(source) == manifest_fingerprint(
            build_evaluation_manifest(source)
        )

    def test_runtime_junk_does_not_change_the_stamp(self, tmp_path):
        """Both halves ignore runtime junk, so an EXECUTED tree — maintainer
        calibration imports the suite before registration — still hashes to
        the identity archives are stamped with. When the framework side
        included junk, the registered head could never equal any stamp
        (observed live)."""
        source = make_evaluation_source(tmp_path)
        clean = sandbox.fingerprint_tree(source)
        pycache = source / "__pycache__"
        pycache.mkdir()
        (pycache / "grader.cpython-312.pyc").write_bytes(b"\x00compiled")
        (source / "stray.pyc").write_bytes(b"\x00stray")
        assert sandbox.fingerprint_tree(source) == clean
        assert (
            manifest_fingerprint(build_evaluation_manifest(source)) == clean
        )

    def test_empty_tree_raises(self, tmp_path):
        empty = tmp_path / "kapso_evaluation"
        empty.mkdir()
        with pytest.raises(ValueError, match="no files"):
            sandbox.fingerprint_tree(empty)


class TestSnapshotAndAllocation:
    def test_snapshot_is_junk_free_idempotent_and_rehashes_to_its_name(
        self, tmp_path
    ):
        source = make_evaluation_source(tmp_path)
        (source / "__pycache__").mkdir()
        (source / "__pycache__" / "x.pyc").write_bytes(b"junk")
        head = sandbox.fingerprint_tree(source)
        root = tmp_path / "work"
        first = sandbox.snapshot_evaluator_tree(root, source, head)
        assert not (first / "__pycache__").exists()
        assert manifest_fingerprint(build_evaluation_manifest(first)) == head
        (source / "kapso_eval.py").write_text("changed after snapshot")
        again = sandbox.snapshot_evaluator_tree(root, source, head)
        assert again == first
        assert manifest_fingerprint(build_evaluation_manifest(first)) == head

    def test_run_dirs_allocate_sequentially_after_existing(self, tmp_path):
        runs_root = tmp_path / "runs"
        (runs_root / "run_0007" / "private").mkdir(parents=True)
        allocated = [sandbox.allocate_run_dir(runs_root).name for _ in range(3)]
        assert allocated == ["run_0008", "run_0009", "run_0010"]
        assert (runs_root / "run_0009" / "private").is_dir()


class TestSelectionLabels:
    def test_finalize_promotes_of_record_and_supersedes_siblings(self, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir()
        make_run(runs, "run_0001", session="s1", score=0.5, evaluator_id="v1")
        make_run(runs, "run_0002", session="s1", score=0.6, evaluator_id="v1")
        make_run(runs, "run_0003", session="s2", score=0.7, evaluator_id="v1")
        archive.finalize_session_run(
            runs, {"fidelity": "full", "run": "run_0002", "session": "s1"}, True
        )
        statuses = {
            n: json.loads((runs / n / "private" / "selection.json").read_text())[
                "status"
            ]
            for n in ("run_0001", "run_0002", "run_0003")
        }
        assert statuses == {
            "run_0001": "superseded",
            "run_0002": "final",
            "run_0003": "pending",
        }

    def test_judge_veto_marks_invalid_and_self_void_outranks_promotion(
        self, tmp_path
    ):
        runs = tmp_path / "runs"
        runs.mkdir()
        make_run(runs, "run_0001", session="s1", score=0.5, evaluator_id="v1")
        archive.finalize_session_run(
            runs, {"fidelity": "full", "run": "run_0001", "session": "s1"}, False
        )
        assert (
            json.loads(
                (runs / "run_0001" / "private" / "selection.json").read_text()
            )["status"]
            == "invalid"
        )
        make_run(
            runs,
            "run_0002",
            session="s1",
            score=0.9,
            evaluator_id="v1",
            status="self-voided",
        )
        archive.finalize_session_run(
            runs, {"fidelity": "full", "run": "run_0002", "session": "s1"}, True
        )
        assert (
            json.loads(
                (runs / "run_0002" / "private" / "selection.json").read_text()
            )["status"]
            == "self-voided"
        )

    def test_missing_run_name_in_full_manifest_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no run name"):
            archive.finalize_session_run(
                tmp_path, {"fidelity": "full", "run": "", "session": "s1"}, True
            )

    def test_infer_session_finals_last_run_wins_and_decided_labels_stay(
        self, tmp_path
    ):
        runs = tmp_path / "runs"
        runs.mkdir()
        make_run(runs, "run_0001", session="s1", score=0.5, evaluator_id="v1")
        make_run(runs, "run_0002", session="s1", score=0.6, evaluator_id="v1")
        make_run(
            runs,
            "run_0003",
            session="s2",
            score=0.7,
            evaluator_id="v1",
            status="self-voided",
        )
        archive.infer_session_finals(runs)
        statuses = {
            n: json.loads((runs / n / "private" / "selection.json").read_text())[
                "status"
            ]
            for n in ("run_0001", "run_0002", "run_0003")
        }
        assert statuses == {
            "run_0001": "superseded",
            "run_0002": "final",
            "run_0003": "self-voided",
        }

    def test_void_stamps_reason_and_refuses_cross_session(self, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir()
        make_run(runs, "run_0001", session="s1", score=0.5, evaluator_id="v1")
        record = sandbox.void_run(
            runs, "run_0001", session="s1", reason="leaky feature"
        )
        assert record["status"] == "self-voided"
        assert record["reason"] == "leaky feature"
        make_run(runs, "run_0002", session="s2", score=0.6, evaluator_id="v1")
        with pytest.raises(PermissionError, match="belongs to session"):
            sandbox.void_run(runs, "run_0002", session="s1", reason="not mine")
        with pytest.raises(ValueError, match="non-empty"):
            sandbox.void_run(runs, "run_0001", session="s1", reason="  ")


class TestHeadResolution:
    def test_head_is_the_stamp_of_the_newest_archive(self, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir()
        make_run(runs, "run_0001", session="s1", score=0.5, evaluator_id="v1")
        make_run(runs, "run_0002", session="s1", score=0.6, evaluator_id="v2")
        assert archive.resolve_head(runs) == "v2"

    def test_unstamped_newest_archive_raises(self, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir()
        run_dir = make_run(
            runs, "run_0001", session="s1", score=0.5, evaluator_id="v1"
        )
        label = run_dir / "private" / "selection.json"
        record = json.loads(label.read_text())
        del record["evaluator_id"]
        label.write_text(json.dumps(record))
        with pytest.raises(ValueError, match="no evaluator stamp"):
            archive.resolve_head(runs)

    def test_empty_archive_raises(self, tmp_path):
        (tmp_path / "runs").mkdir()
        with pytest.raises(FileNotFoundError, match="no runs"):
            archive.resolve_head(tmp_path / "runs")

    def test_missing_selection_label_raises(self, tmp_path):
        runs = tmp_path / "runs"
        (runs / "run_0001" / "private").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="pre-label"):
            archive.resolve_head(runs)


class TestSelectFinal:
    def test_pools_head_finals_only_and_ships_the_max(self, governed_archive):
        root, head, _ = governed_archive
        runs = root / "runs"
        make_run(runs, "run_0001", session="s2", score=0.9, evaluator_id="old")
        make_run(runs, "run_0002", session="s1", score=0.6, evaluator_id=head)
        make_run(
            runs,
            "run_0003",
            session="s1",
            score=0.8,
            evaluator_id=head,
            # Evolved evaluators mix provenance labels into metrics; the
            # selection keeps numerics (incl. numeric strings) and drops
            # labels instead of crashing (live: user-ignore, 2026-08-12).
            metrics={
                "auroc": 0.8,
                "f1": 0.5,
                "protocol": "weekly_origin_mean_roc_auc_v1",
                "n_windows": "12",
            },
        )
        make_run(
            runs,
            "run_0004",
            session="s3",
            score=0.95,
            evaluator_id=head,
            status="self-voided",
        )
        result = archive.select_final(root, higher_is_better=True)
        assert result.head_evaluator_id == head
        # run_0001 is s2's final but measured under a superseded evaluator:
        # excluded, even though its 0.9 beats every head score
        assert result.winner_run == "run_0003"
        assert result.winner_score == pytest.approx(0.8)
        assert result.winner_metrics == {"auroc": 0.8, "f1": 0.5, "n_windows": 12.0}
        assert "run_0001" in result.excluded
        assert "never re-ranked across rulers" in result.excluded["run_0001"]
        # superseded and voided runs were never rescored at all
        assert set(result.scored) == {"run_0003"}

    def test_direction_lower_is_better(self, governed_archive):
        root, head, _ = governed_archive
        runs = root / "runs"
        make_run(runs, "run_0001", session="s1", score=0.30, evaluator_id=head)
        make_run(runs, "run_0002", session="s2", score=0.12, evaluator_id=head)
        result = archive.select_final(root, higher_is_better=False)
        assert result.winner_run == "run_0002"
        assert result.winner_score == pytest.approx(0.12)

    def test_edited_archive_score_trips_the_wire(self, governed_archive):
        root, head, _ = governed_archive
        runs = root / "runs"
        run_dir = make_run(
            runs, "run_0001", session="s1", score=0.6, evaluator_id=head
        )
        stored = run_dir / "manifest.txt"
        stored.write_text(stored.read_text().replace("0.6", "0.99"))
        with pytest.raises(ValueError, match="does not match its recomputation"):
            archive.select_final(root, higher_is_better=True)

    def test_tampered_snapshot_is_rejected_before_scoring(self, governed_archive):
        root, head, _ = governed_archive
        runs = root / "runs"
        make_run(runs, "run_0001", session="s1", score=0.6, evaluator_id=head)
        snapshot = root / "evaluators" / head
        (snapshot / "grader.py").write_bytes(b"return 1.0  # reward hack\n")
        with pytest.raises(ValueError, match="snapshot was modified"):
            archive.select_final(root, higher_is_better=True)

    def test_provided_byte_anchor_mismatch_raises(self, governed_archive):
        root, head, _ = governed_archive
        runs = root / "runs"
        make_run(runs, "run_0001", session="s1", score=0.6, evaluator_id=head)
        good = hashlib.sha256(PROVIDED_GRADER_BYTES).hexdigest()
        result = archive.select_final(
            root, higher_is_better=True, provided_files={"grader.py": good}
        )
        assert result.winner_run == "run_0001"
        with pytest.raises(ValueError, match="does not match the shipped copy"):
            archive.select_final(
                root,
                higher_is_better=True,
                provided_files={"grader.py": "0" * 64},
            )

    def test_expected_head_mismatch_raises(self, governed_archive):
        root, head, _ = governed_archive
        make_run(
            root / "runs", "run_0001", session="s1", score=0.6, evaluator_id=head
        )
        with pytest.raises(ValueError, match="expected registered head"):
            archive.select_final(
                root, higher_is_better=True, expected_head_id="f" * 64
            )

    def test_no_head_finals_returns_empty_selection(self, governed_archive):
        """Absence of results is an outcome to report, not corruption."""
        root, head, _ = governed_archive
        runs = root / "runs"
        make_run(runs, "run_0001", session="s1", score=0.9, evaluator_id="old")
        make_run(
            runs,
            "run_0002",
            session="s2",
            score=0.8,
            evaluator_id=head,
            status="self-voided",
        )
        result = archive.select_final(root, higher_is_better=True)
        assert result.winner_run is None
        assert result.winner_score is None
        assert result.scored == {}
        assert "run_0001" in result.excluded

    def test_missing_snapshot_raises(self, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir()
        make_run(runs, "run_0001", session="s1", score=0.6, evaluator_id="v9")
        with pytest.raises(FileNotFoundError, match="no evaluator snapshot"):
            archive.select_final(tmp_path, higher_is_better=True)


def test_maintainer_prompts_carry_the_rescore_contract():
    """Every entrypoint the maintainer builds or evolves must support
    --rescore with a shared scoring path — final selection ranks archives
    exclusively through it, so a wrapper that dropped the mode would make
    every final unmeasurable at the last mile."""
    prompts_dir = (
        Path(__file__).parents[1]
        / "src/kapso/execution/evaluation_maintainer/prompts"
    )
    for name in ("setup_provided.md", "setup_build.md", "change_request.md"):
        text = (prompts_dir / name).read_text()
        assert "--rescore" in text, f"{name} lost the rescore contract"
        assert "scoring path" in text, (
            f"{name} lost the shared-scoring-path requirement"
        )
    triage = (prompts_dir / "change_request.md").read_text()
    # The referee must be able to ACCEPT the request the instructions teach
    # agents to file: a statistically evidenced mismeasurement is a defect
    # class, not lobbying. Without this the whole channel dead-ends.
    assert "Measurement validity" in triage
    assert "standard errors" in triage
    assert "representativeness" in triage.lower()
    # A score-redefining wrapper owns the archived manifest line, or every
    # post-transition final trips the tamper wire at select_final.
    assert "manifest.txt" in triage
    assert "ONE manifest line" in triage
    # Migration tiers: the first live transition (rel-event/user-ignore,
    # 2026-08-09) rebuilt the wrapper candidate-aware and 5 of 6 frontier
    # designs crashed under it — the bridge could carry one node. The
    # maintainer must implement the lowest tier so prior archives stay
    # measurable across a version change.
    assert "LOWEST migration tier" in triage
    assert "UNCHANGED entrypoint" in triage
    assert "Tier 3, contract-breaking" in triage
    for name in ("setup_provided.md", "setup_build.md"):
        text = (prompts_dir / name).read_text()
        assert "Design for future protocol changes" in text, (
            f"{name} lost the migration-friendly design requirement"
        )
        assert "raw outputs (not just the score)" in text


def test_evaluation_instructions_demand_early_filing_and_low_tiers():
    """The requester-side half of the migration contract. The live c2 test
    showed the agent diagnosing the defect in iteration 1 but filing at
    5h16m of a 6h cap: the transition then voided 12 archives with no
    budget left to exploit the corrected metric. The instructions must
    force the filing to the first confirmation, demand the least-breaking
    remedy, and point post-transition sessions at porting voided designs."""
    source = (
        Path(__file__).parents[1]
        / "src/kapso/execution/search_strategies/generic/registered_evaluation.py"
    ).read_text()
    assert "TIMING — file at the FIRST confirmation" in source
    assert "REMEDY SHAPE — propose the least-breaking remedy" in source
    assert "scores never cross evaluator versions" in source
    assert "porting the strongest of them to the current evaluation" in source


def test_vendored_copies_are_byte_identical():
    """Benchmark suites vendor the sandbox contract; a drifted copy would
    stamp archives with a fingerprint no framework code can reproduce."""
    repo_root = Path(__file__).parents[1]
    master = repo_root / "src/kapso/execution/evaluation_archive_sandbox.py"
    vendored = [
        repo_root / "benchmarks/relbench/data/generic_eval/kapso_eval_archive.py",
    ]
    for copy in vendored:
        assert copy.read_bytes() == master.read_bytes(), (
            f"{copy} has drifted from the framework master — recopy it"
        )
