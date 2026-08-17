# Scorecard + gauntlet-harness tests (P3).
#
# Each test names the regression it catches (Rule 9).

import pytest

from kapso.learning.graders.gauntlet import (
    assemble_gauntlet,
    build_duplicate_fixture,
    substance_diff,
)
from kapso.learning.graders.scorecard import (
    aggregate,
    calibration_table,
    paired_deltas,
    validate_verdict,
)
from tests.test_bank_retriever import build_bank, card_text

GRADERS_CONFIG = {"calibration_min": 4, "calibration_buckets": [0.4, 0.7]}


def report(trajectory, foresight, accuracy=0.8, serving=0.5, split_version=1):
    return {
        "trajectory": trajectory,
        "split_version": split_version,
        "hindcast": {"foresight": foresight, "accuracy": accuracy,
                     "serving": serving},
    }


def test_aggregate_shows_distribution_and_nulls():
    # Regression (§2.1): mean ± SE with per-trajectory values and null counts
    # — nulls excluded from the mean, never imputed.
    reports = [report("t1", 0.4), report("t2", 0.6), report("t3", None)]
    agg = aggregate(reports)
    assert agg["foresight"]["values"] == [0.4, 0.6]
    assert agg["foresight"]["mean"] == pytest.approx(0.5)
    assert agg["foresight"]["n"] == 2 and agg["foresight"]["nulls"] == 1
    assert agg["foresight"]["se"] is not None


def test_paired_deltas_math():
    # Regression: the paired statistic — per-trajectory deltas, mean ± SE.
    candidate = [report("t1", 0.5), report("t2", 0.7)]
    incumbent = [report("t1", 0.4), report("t2", 0.5)]
    deltas = paired_deltas(candidate, incumbent)
    assert deltas["n_pairs"] == 2
    assert deltas["foresight"]["deltas"] == [0.1, 0.2]
    assert deltas["foresight"]["mean"] == pytest.approx(0.15)


def test_paired_comparison_requires_same_exam():
    # Regression (§2.1/§3): cross-split pairing or mismatched trajectory sets
    # invalidate the comparison — refuse, never fudge.
    with pytest.raises(ValueError, match="identical trajectory sets"):
        paired_deltas([report("t1", 0.5)], [report("t2", 0.4)])
    with pytest.raises(ValueError, match="split versions"):
        paired_deltas([report("t1", 0.5, split_version=1)],
                      [report("t1", 0.4, split_version=2)])


def test_calibration_pools_by_claimed_bucket():
    # Regression (§2.2): realized agreement per CLAIMED bucket; absent below
    # the pooling minimum.
    settlements = [
        {"claimed": 0.8, "agreed": True}, {"claimed": 0.9, "agreed": True},
        {"claimed": 0.5, "agreed": False}, {"claimed": 0.2, "agreed": True},
    ]
    table = calibration_table(settlements, GRADERS_CONFIG)
    assert table["pooled"] == 4
    high = next(b for b in table["buckets"] if b["claimed"] == "[0.7–1.0)")
    assert high["n"] == 2 and high["agreed"] == 2
    assert calibration_table(settlements[:2], GRADERS_CONFIG) is None


def test_verdict_gates_dominate_and_numbers_recomputed():
    # Regression (§2.3/§6.6): a gauntlet FAIL can never be accepted, and an
    # agent-quoted delta must equal the frame's arithmetic.
    deltas = {"foresight": {"mean": 0.15}, "accuracy": {"mean": 0.0},
              "serving": {"mean": 0.02}}
    findings = validate_verdict(
        {"decision": "accept", "rationale": "solid", "foresight_delta": 0.15},
        deltas, "FAIL",
    )
    assert any("gates dominate scores" in f for f in findings)
    findings = validate_verdict(
        {"decision": "accept", "rationale": "solid", "foresight_delta": 0.20},
        deltas, "PASS",
    )
    assert any("never trusted for arithmetic" in f for f in findings)
    findings = validate_verdict(
        {"decision": "within-noise", "rationale": "SE swamps the deltas"},
        deltas, "PASS",
    )
    assert findings == []


def test_verdict_no_naked_tags():
    # Regression: decision without rationale rejects.
    findings = validate_verdict({"decision": "reject", "rationale": " "},
                                {}, "PASS")
    assert any("no naked tags" in f for f in findings)


def test_substance_diff_stability_semantics(tmp_path):
    # Regression (§2.3): substance = card set, state, version, scores within
    # tolerance; prose differences are NOT substance.
    bank_a = build_bank(tmp_path / "a", {
        "same-card": card_text("same-card", score=0.60),
        "only-a-card": card_text("only-a-card"),
    })
    bank_b = build_bank(tmp_path / "b", {
        "same-card": card_text("same-card", score=0.75,
                               body="Different prose, same substance? No — "
                                    "score moved beyond tolerance [E1]."),
    })
    differences = substance_diff(str(bank_a), str(bank_b), tolerance=0.10)
    assert any("only-a-card exists only in run A" in d for d in differences)
    assert any("exceeds tolerance" in d for d in differences)

    bank_c = build_bank(tmp_path / "c", {
        "same-card": card_text("same-card", score=0.65,
                               body="Reworded prose entirely [E1].")})
    bank_d = build_bank(tmp_path / "d", {
        "same-card": card_text("same-card", score=0.60)})
    assert substance_diff(str(bank_c), str(bank_d), tolerance=0.10) == []


def test_duplicate_fixture_clones_under_fresh_identity(tmp_path):
    # Regression: the duplicate trap's mechanical base — same content, new
    # trajectory identity.
    source = tmp_path / "mined"
    source.mkdir()
    (source / "index.md").write_text("# campaign\n")
    (source / "strategy.md").write_text("lens story\n")
    target = build_duplicate_fixture(
        str(source), str(tmp_path / "fixtures"),
        "rel-clone--user-churn/20260102T000000_lane-x1",
    )
    assert (target / "index.md").read_text() == "# campaign\n"
    assert "rel-clone--user-churn" in str(target)


def test_gauntlet_assembly_rolls_fail_and_demands_rationales(tmp_path):
    # Regression: any trap FAIL rolls the verdict to FAIL; naked tags refuse
    # to assemble.
    context = {
        "learner_version": "crew_v1", "bank_head": "lr_x", "batch": [],
        "rolled_rationale": "stability alone rejects this version",
        "duplicate_proof": "fixture: fixtures/duplicate/",
        "stability_proof": "diff: diffs/stability.patch",
    }
    text = assemble_gauntlet(
        {"duplicate": {"verdict": "PASS", "rationale": "byte-empty diff"},
         "stability": {"verdict": "FAIL", "rationale": "card born in B only"}},
        context,
    )
    assert "verdict: FAIL" in text
    assert "## duplicate — construction + proof" in text
    with pytest.raises(ValueError, match="no rationale"):
        assemble_gauntlet(
            {"duplicate": {"verdict": "PASS", "rationale": ""}}, context
        )
