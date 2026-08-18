# A/B verdict math tests (P6.4, GS§2.5): paired deltas oriented by metric
# direction, the thin-task guard KPI, and the gate semantics — not-run is
# never passed, required + within-noise blocks, noise is never rounded to a
# win (Rule 9: each is a promotion-integrity regression).

from kapso.learning.ab import ab_verdict

AB_REQUIRED = {"required": True, "pairs": 5}
AB_WAIVED = {"required": False, "pairs": 5}


def pair(task, candidate, incumbent, maximize=True, thin=False):
    return {"task": task, "candidate": candidate, "incumbent": incumbent,
            "maximize": maximize, "thin": thin}


def test_not_run_is_recorded_never_passed():
    waived = ab_verdict([], AB_WAIVED)
    assert waived["verdict"] == "not-run" and not waived["blocking"]
    required = ab_verdict([], AB_REQUIRED)
    assert required["verdict"] == "not-run" and required["blocking"]


def test_clear_win_and_metric_orientation():
    result = ab_verdict([
        pair("t1", 0.72, 0.70),
        pair("t2", 0.71, 0.695),
        # minimize-metric task: lower candidate value is a positive delta
        pair("t3", 0.30, 0.32, maximize=False),
        pair("t4", 0.68, 0.665),
    ], AB_REQUIRED)
    assert result["verdict"] == "win" and not result["blocking"]
    assert all(row["delta"] > 0 for row in result["deltas"])
    assert result["mean"] > result["se"] > 0


def test_within_noise_blocks_when_required_and_not_when_waived():
    pairs = [
        pair("t1", 0.700, 0.701),
        pair("t2", 0.702, 0.700),
        pair("t3", 0.699, 0.700),
        pair("t4", 0.701, 0.699),
    ]
    required = ab_verdict(pairs, AB_REQUIRED)
    assert required["verdict"] == "within-noise" and required["blocking"]
    waived = ab_verdict(pairs, AB_WAIVED)
    assert waived["verdict"] == "within-noise" and not waived["blocking"]


def test_thin_task_regression_trips_the_guard():
    # The mean is a healthy win, but a thin task regressed beyond noise —
    # "never worse where irrelevant" dominates the average.
    result = ab_verdict([
        pair("t1", 0.75, 0.70),
        pair("t2", 0.74, 0.70),
        pair("t3", 0.60, 0.70, thin=True),
        pair("t4", 0.73, 0.70),
    ], AB_WAIVED)
    assert result["verdict"] == "regression"
    assert result["guard"]["regressions"] == ["t3"]
    assert result["guard"]["thin_pairs"] == 1


def test_single_pair_never_certifies_a_win():
    result = ab_verdict([pair("t1", 0.75, 0.70)], AB_REQUIRED)
    assert result["verdict"] == "within-noise" and result["blocking"]
    assert result["se"] is None
