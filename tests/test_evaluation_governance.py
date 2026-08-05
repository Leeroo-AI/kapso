"""Integration tests for the evaluation-governance chain.

This machinery decides what happens when the thing that produces scores
changes underneath a search: candidate tampering must be caught, and a
sanctioned evaluator change must not let old scores be compared against new
ones. It has never fired in a live campaign — 938 archived measurements all
sit under a single evaluator — so these tests are the only thing pinning the
contract down.

Each test asserts a fail-loud path or a documented invariant, not a mock's
own behaviour: hashes are computed over real files on disk, and score
projection is exercised through the real dataclasses.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from kapso.execution.evaluation_integrity import (
    build_evaluation_manifest,
    manifest_fingerprint,
    verify_data_manifest,
    verify_evaluation_tree,
)
from kapso.execution.fidelity import (
    ComparabilityClass,
    EvaluationAttempt,
    attempts_in_class,
    project_score,
)

GRADER = "grader.py"
GRADER_SRC = "def score(pred, labels):\n    return float((pred == labels).mean())\n"


def _eval_tree(root: Path) -> Path:
    """A minimal evaluation tree standing in for kapso_evaluation/."""
    d = root / "kapso_evaluation"
    d.mkdir(parents=True)
    (d / GRADER).write_text(GRADER_SRC)
    (d / "README.md").write_text("registered evaluation entrypoint\n")
    return d


class TestEvaluatorTamperingIsCaught:
    """A candidate must not be able to alter what scores it."""

    def test_unmodified_tree_verifies(self, tmp_path):
        d = _eval_tree(tmp_path)
        manifest = build_evaluation_manifest(d)
        report = verify_evaluation_tree(d, manifest)
        assert report.valid is True
        assert report.error == ""
        assert report.fingerprint == manifest_fingerprint(manifest)

    def test_edited_grader_is_rejected(self, tmp_path):
        d = _eval_tree(tmp_path)
        manifest = build_evaluation_manifest(d)
        # the reward hack: make the grader return a constant
        (d / GRADER).write_text("def score(pred, labels):\n    return 1.0\n")
        report = verify_evaluation_tree(d, manifest)
        assert report.valid is False
        assert GRADER in report.error

    def test_deleted_file_is_rejected(self, tmp_path):
        d = _eval_tree(tmp_path)
        manifest = build_evaluation_manifest(d)
        (d / GRADER).unlink()
        report = verify_evaluation_tree(d, manifest)
        assert report.valid is False
        assert GRADER in report.error

    def test_smuggled_in_source_is_rejected(self, tmp_path):
        """A NEW .py the manifest never sanctioned is still evaluator source."""
        d = _eval_tree(tmp_path)
        manifest = build_evaluation_manifest(d)
        (d / "sneaky_scorer.py").write_text("SCORE = 1.0\n")
        report = verify_evaluation_tree(d, manifest)
        assert report.valid is False
        assert "sneaky_scorer.py" in report.error

    def test_fingerprint_changes_with_content(self, tmp_path):
        """evaluator_id is the manifest fingerprint: content change => new id."""
        d = _eval_tree(tmp_path)
        first = manifest_fingerprint(build_evaluation_manifest(d))
        (d / GRADER).write_text(GRADER_SRC + "# tuned\n")
        assert manifest_fingerprint(build_evaluation_manifest(d)) != first


class TestProtectedDataTamperingIsCaught:
    """The inputs half of evaluation identity — scoring yourself on rewritten
    labels is the failure this guard was added for."""

    def test_untouched_data_passes(self, tmp_path):
        (tmp_path / "labels.parquet").write_bytes(b"\x00label-bytes")
        manifest = {
            k: v
            for k, v in build_evaluation_manifest(tmp_path).items()
            if k == "labels.parquet"
        }
        assert manifest, "expected the data file in the manifest"
        assert verify_data_manifest(tmp_path, manifest) == ""

    def test_rewritten_labels_are_reported(self, tmp_path):
        (tmp_path / "labels.parquet").write_bytes(b"\x00label-bytes")
        manifest = build_evaluation_manifest(tmp_path)
        (tmp_path / "labels.parquet").write_bytes(b"\x01all-true")
        problem = verify_data_manifest(tmp_path, manifest)
        assert problem.startswith("Protected evaluation data changed")
        assert "modified:labels.parquet" in problem

    def test_deleted_data_is_reported(self, tmp_path):
        (tmp_path / "labels.parquet").write_bytes(b"\x00label-bytes")
        manifest = build_evaluation_manifest(tmp_path)
        (tmp_path / "labels.parquet").unlink()
        assert "missing:labels.parquet" in verify_data_manifest(tmp_path, manifest)


class TestScoresNeverCrossEvaluators:
    """The core invariant: no score crosses an evaluator_id boundary."""

    @staticmethod
    def _node(*attempts):
        return SimpleNamespace(evaluation_attempts=list(attempts))

    @staticmethod
    def _attempt(evaluator_id, score, fidelity="full", fraction=1.0, seed=1337):
        return EvaluationAttempt(
            commit_sha="deadbeef",
            evaluator_id=evaluator_id,
            fidelity=fidelity,
            fraction=fraction,
            seed=seed,
            score=score,
        )

    def test_old_score_is_invisible_under_the_new_evaluator(self):
        node = self._node(self._attempt("v1", 0.95))
        new = ComparabilityClass(
            evaluator_id="v2", fidelity="full", fraction=1.0, seed=1337
        )
        assert attempts_in_class(node, new) == []
        # None, not the stale 0.95 — an unmeasured node must not win a ranking
        assert project_score(node, new) is None

    def test_score_projects_within_its_own_class(self):
        node = self._node(self._attempt("v1", 0.90), self._attempt("v1", 0.80))
        cls = ComparabilityClass(
            evaluator_id="v1", fidelity="full", fraction=1.0, seed=1337
        )
        assert project_score(node, cls) == pytest.approx(0.85)

    def test_fidelity_also_separates_classes(self):
        """A fast measurement is not comparable to a full one."""
        node = self._node(self._attempt("v1", 0.99, fidelity="fast", fraction=0.15))
        full = ComparabilityClass(
            evaluator_id="v1", fidelity="full", fraction=1.0, seed=1337
        )
        assert project_score(node, full) is None

    def test_transition_leaves_every_node_unmeasured(self):
        """After an evaluator change the whole frontier must be re-measured;
        nothing may inherit its old ranking."""
        frontier = [self._node(self._attempt("v1", s)) for s in (0.7, 0.8, 0.9)]
        new = ComparabilityClass(
            evaluator_id="v2", fidelity="full", fraction=1.0, seed=1337
        )
        assert all(project_score(n, new) is None for n in frontier)


class TestComparabilityClassValidation:
    """Fail loud on a malformed ruler rather than silently mis-comparing."""

    def test_empty_evaluator_id_raises(self):
        with pytest.raises(ValueError, match="evaluator_id"):
            ComparabilityClass(
                evaluator_id="", fidelity="full", fraction=1.0, seed=1337
            )

    def test_unknown_fidelity_raises(self):
        with pytest.raises(ValueError, match="fidelity"):
            ComparabilityClass(
                evaluator_id="v1", fidelity="medium", fraction=1.0, seed=1337
            )

    def test_out_of_range_fraction_raises(self):
        with pytest.raises(ValueError, match="fraction"):
            ComparabilityClass(
                evaluator_id="v1", fidelity="full", fraction=0.0, seed=1337
            )
