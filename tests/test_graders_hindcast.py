# Hindcast + split tests — the P3 contract (docs/plans/learning/p3-grader-suite.md).
#
# The canonical fixture is the grader-scoring doc's §6 worked example: its
# corridor arithmetic (foresight 0.45 admitted on center 0.40; serving 0.40
# admitted / 0.50 rejected on center ~0.22; overall span [0.20, 1.00]) is
# reproduced exactly. Each test names the regression it catches (Rule 9).

import pytest

from kapso.learning.graders.hindcast import (
    HindcastReport,
    HindcastValidator,
    corridor_centers,
)
from kapso.learning.graders.split import (
    assert_batch_disjoint,
    load_split,
    validate_split,
)

GRADERS_CONFIG = {"score_band": 0.20, "min_settlements": 2}

KNOWN_CARDS = {
    "grouped-rows-fold-leakage", "thin-history-blind-spot", "recency-window",
    "forward-gate", "cross-family-ensemble-candidate", "cold-user-imputation",
}
KNOWN_REFS = {
    "mined/it-1/flow-2.md", "mined/it-2/flow-3.md", "mined/it-3/flow-1.md",
    "mined/it-3/flow-4.md", "mined/it-4/flow-2.md", "learn/mined/it-3/flow-1.md",
}


def worked_example(foresight=0.45, accuracy=0.80, serving=0.40, score=0.55,
                   rationale="Extraction gap binds the overall; novel share 2/7."):
    """The §6 worked example: 2 hits, 3 uncarded, 2 novel; 4+1 settled + THIN;
    one of each serving marker."""
    scores = {
        "foresight": "null" if foresight is None else foresight,
        "accuracy": "null" if accuracy is None else accuracy,
        "serving": "null" if serving is None else serving,
        "score": "null" if score is None else score,
    }
    return f"""---
trajectory: rel-hm--user-churn/20260819T030000_lane-a1
bank_head: lr_20260817T2100
brief: brief.md
hindcast:
  foresight: {scores['foresight']}
  accuracy: {scores['accuracy']}
  serving: {scores['serving']}
  score: {scores['score']}
  rationale: >-
    {rationale}
---

## Extraction
- **HIT-SERVED** — grouped-rows fold leakage: re-derived despite the brief
  [mined/it-2/flow-3.md#evaluation].
- **HIT-SERVED** — thin-history boundary: banked and served
  [mined/it-1/flow-2.md#evaluation].
- **MISS-UNCARDED** — cross-family ensemble gating: learn-set source
  [learn/mined/it-3/flow-1.md#evaluation]; never carded.
- **MISS-UNCARDED** — depth-slice profiling: learn-set source
  [learn/mined/it-3/flow-1.md#judgment].
- **MISS-UNCARDED** — seed-pinning discipline: learn-set source
  [learn/mined/it-3/flow-1.md#implementation].
- **MISS-NOVEL** — sparse-session recency collapse: no learn-set source
  (searched family-wide; attested) [mined/it-3/flow-4.md].
- **MISS-NOVEL** — holiday-window drift: no learn-set source (attested)
  [mined/it-3/flow-4.md].

## Claims settlement
- **CONTRADICTED** — [insight: recency-window]: predicts gain; measured
  −0.004 ± 0.001 [mined/it-3/flow-1.md#evaluation].
- **AGREED** — [insight: thin-history-blind-spot]: predicted degradation;
  measured −0.021 ± 0.006 exactly there [mined/it-1/flow-2.md#evaluation].
- **AGREED** — [insight: grouped-rows-fold-leakage]: +0.006 ± 0.002
  [mined/it-2/flow-3.md#evaluation].
- **AGREED** — [insight: forward-gate]: +0.004 ± 0.001
  [mined/it-4/flow-2.md#evaluation].
- **AGREED** — [insight: cold-user-imputation]: +0.003 ± 0.001
  [mined/it-4/flow-2.md#evaluation].
- **THIN** — [procedure: forward-gate]: directional, under significance
  (+0.001 ± 0.002) [mined/it-4/flow-2.md#evaluation].

## Serving
- **SERVED-USED** — [insight: thin-history-blind-spot]: served, cited, steered
  the design [mined/it-1/flow-2.md].
- **UPTAKE-FAIL** — [insight: grouped-rows-fold-leakage]: served, never cited,
  re-derived verbatim [mined/it-2/flow-3.md].
- **SERVE-MISS** — [procedure: cross-family-ensemble-candidate]: banked,
  relevant, below budget [mined/it-4/flow-2.md].
- **SERVE-NOISE** — [insight: cold-user-imputation]: served on a tag match; no
  cold segment existed [mined/it-3/flow-1.md].
"""


def make_validator():
    return HindcastValidator(
        GRADERS_CONFIG,
        ref_exists=lambda path: path in KNOWN_REFS,
        known_cards=KNOWN_CARDS,
    )


def parse(text):
    return HindcastReport.parse(text)


def test_worked_example_admits(tmp_path):
    # Regression: the §6 example is the canonical admitted report — corridor
    # arithmetic must reproduce exactly.
    report = parse(worked_example())
    centers = corridor_centers(report.counts())
    assert centers["foresight"] == pytest.approx(2 / 5)
    assert centers["accuracy"] == pytest.approx(4 / 5)
    assert centers["serving"] == pytest.approx((1 / 3) * (2 / 3))
    assert make_validator().validate(report) == []


def test_generous_serving_score_rejected(tmp_path):
    # Regression: the corridor is the honesty tether — §6's 0.50-serving case
    # bounces with the corridor finding.
    report = parse(worked_example(serving=0.50))
    findings = make_validator().validate(report)
    assert any("`serving` 0.50 is outside its corridor" in f for f in findings)


def test_overall_must_stay_in_dimension_span(tmp_path):
    # Regression: the overall may not escape its own dimensions' span ± band.
    report = parse(worked_example(score=0.05))
    findings = make_validator().validate(report)
    assert any("escapes the span" in f for f in findings)


def test_null_is_a_verdict_not_a_gap(tmp_path):
    # Regression (§0.2): a number over an empty base and a null over a
    # non-empty base are both rejected.
    report = parse(worked_example(foresight=None))
    findings = make_validator().validate(report)
    assert any("`foresight` is null but its evidence base is non-empty" in f
               for f in findings)


def test_false_precision_rejected(tmp_path):
    # Regression (§0.1): more than two decimals is noise, rejected.
    report = parse(worked_example(accuracy=0.803))
    findings = make_validator().validate(report)
    assert any("more than two decimals" in f for f in findings)


def test_unknown_marker_and_unknown_card_reject(tmp_path):
    # Regression: the marker vocabulary and the card namespace are closed.
    text = worked_example().replace("**SERVE-NOISE**", "**SERVE-JUNK**")
    findings = make_validator().validate(parse(text))
    assert any("SERVE-JUNK" in f for f in findings)

    text = worked_example().replace("[insight: recency-window]",
                                    "[insight: invented-card]")
    findings = make_validator().validate(parse(text))
    assert any("invented-card" in f for f in findings)


def test_unresolvable_ref_rejects(tmp_path):
    # Regression: every ref must resolve in the trajectory.
    text = worked_example().replace("mined/it-2/flow-3.md#evaluation",
                                    "mined/it-9/ghost.md#evaluation")
    findings = make_validator().validate(parse(text))
    assert any("mined/it-9/ghost.md" in f for f in findings)


def test_settlements_must_be_liftable(tmp_path):
    # Regression (§4): AGREED/CONTRADICTED without a measured delta cannot
    # lift into evidence.
    text = worked_example().replace("measured\n  −0.004 ± 0.001", "measured a drop")
    findings = make_validator().validate(parse(text))
    assert any("no measured delta" in f for f in findings)


def test_rationale_duties(tmp_path):
    # Regression: no naked scores; the novel share must be stated when
    # MISS-NOVEL entries exist.
    report = parse(worked_example(rationale="Solid bank."))
    findings = make_validator().validate(report)
    assert any("novel share" in f for f in findings)


def test_accuracy_nulls_below_settlement_floor(tmp_path):
    # Regression (§1.3): thin evidence is reported as thin, never scored —
    # under min_settlements the accuracy number itself becomes a finding.
    validator = HindcastValidator(
        {"score_band": 0.20, "min_settlements": 6},
        ref_exists=lambda p: p in KNOWN_REFS,
        known_cards=KNOWN_CARDS,
    )
    findings = validator.validate(parse(worked_example()))
    assert any("`accuracy` carries a number but its evidence base is empty" in f
               for f in findings)


SPLIT_TEXT = """
version: 1
rule: split by (family, time), never by task
rationale: founding split over the D1 corpus
learn:
  - {id: rel-amazon--user-churn/20260101T000000_lane-t1, family: rel-amazon, date: 2026-01-01}
  - {id: rel-hm--user-churn/20260102T000000_lane-t2, family: rel-hm, date: 2026-01-02}
held_out:
  - {id: rel-event--user-repeat/20260103T000000_lane-t3, family: rel-event, date: 2026-01-03}
"""


def write_split(tmp_path, text=SPLIT_TEXT):
    path = tmp_path / "split.yaml"
    path.write_text(text)
    return str(path)


def manifest(trajectory_id, dataset):
    return {"id": trajectory_id, "dataset": dataset}


STORE_MANIFESTS = [
    manifest("rel-amazon--user-churn/20260101T000000_lane-t1", "rel-amazon"),
    manifest("rel-hm--user-churn/20260102T000000_lane-t2", "rel-hm"),
    manifest("rel-event--user-repeat/20260103T000000_lane-t3", "rel-event"),
]


def test_split_loads_and_validates_clean(tmp_path):
    split = load_split(write_split(tmp_path))
    assert validate_split(split, STORE_MANIFESTS) == []


def test_split_family_on_both_sides_trips(tmp_path):
    # Regression: contamination — a family straddling the split invalidates
    # the exam.
    text = SPLIT_TEXT.replace("family: rel-event", "family: rel-amazon")
    split = load_split(write_split(tmp_path, text))
    findings = validate_split(split, STORE_MANIFESTS)
    assert any("appears on both sides" in f for f in findings)


def test_split_must_cover_store_exactly(tmp_path):
    # Regression: every store trajectory exactly once — an unlisted or
    # phantom trajectory is a finding.
    split = load_split(write_split(tmp_path))
    extra = STORE_MANIFESTS + [manifest("rel-f1--driver-dnf/20260104T000000_lane-t4", "rel-f1")]
    findings = validate_split(split, extra)
    assert any("is not in the split" in f for f in findings)
    findings = validate_split(split, STORE_MANIFESTS[:2])
    assert any("which is not in the store" in f for f in findings)


def test_split_missing_rationale_raises(tmp_path):
    # Regression: a version without its rationale is not a valid exam.
    text = SPLIT_TEXT.replace("rationale: founding split over the D1 corpus\n", "")
    with pytest.raises(ValueError, match="rationale"):
        load_split(write_split(tmp_path, text))


def test_batch_disjointness_twin_check(tmp_path):
    # Regression: the update-run twin — a held-out id in a development batch
    # fails loud before any session exists.
    split = load_split(write_split(tmp_path))
    assert_batch_disjoint(split, ["rel-amazon--user-churn/20260101T000000_lane-t1"])
    with pytest.raises(ValueError, match="held-out trajectories"):
        assert_batch_disjoint(
            split, ["rel-event--user-repeat/20260103T000000_lane-t3"]
        )
