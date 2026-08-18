# Reproduction-gate tests (P7.3, CD§2): decision mismatch fails, metric
# outside the band fails, weak assertions fail validation, and a staged
# fixture output trips actually-invoked (Rule 9: each is a false-green
# replay regression).

from kapso.learning.codify_gates import (
    actually_invoked_findings,
    reproduction_findings,
    weak_assertion_findings,
)

GATES = {
    "decisions": {"gate_cleared": True},
    "metrics": {"validation_delta": {"value": 0.002158, "se": 0.0005}},
    "artifacts": {"predictions": {"path": "outputs/preds.csv", "min_bytes": 10}},
}


def make_workspace(tmp_path, content="a,b\n1,2\n3,4\n"):
    target = tmp_path / "outputs" / "preds.csv"
    target.parent.mkdir(parents=True)
    target.write_text(content)
    return str(tmp_path)


def test_green_reproduction(tmp_path):
    workspace = make_workspace(tmp_path)
    findings = reproduction_findings(
        GATES,
        {"decisions": {"gate_cleared": True},
         "metrics": {"validation_delta": 0.002400}},
        workspace, tolerance_z=2,
    )
    assert findings == []


def test_decision_mismatch_and_metric_band(tmp_path):
    workspace = make_workspace(tmp_path)
    findings = reproduction_findings(
        GATES,
        {"decisions": {"gate_cleared": False},
         "metrics": {"validation_delta": 0.004}},
        workspace, tolerance_z=2,
    )
    assert any("decision outcomes are exact" in f for f in findings)
    assert any("not a reproduction" in f for f in findings)


def test_missing_and_stub_artifacts(tmp_path):
    findings = reproduction_findings(
        GATES,
        {"decisions": {"gate_cleared": True},
         "metrics": {"validation_delta": 0.002158}},
        str(tmp_path), tolerance_z=2,
    )
    assert any("was not produced" in f for f in findings)
    workspace = make_workspace(tmp_path, content="x")
    findings = reproduction_findings(
        GATES,
        {"decisions": {"gate_cleared": True},
         "metrics": {"validation_delta": 0.002158}},
        workspace, tolerance_z=2,
    )
    assert any("a stub, not the outcome" in f for f in findings)


def test_weak_assertions_fail_validation():
    strong = ("assert abs(delta - 0.002158) < band\n"
              "assert outcome['gate_cleared'] is True\n")
    assert weak_assertion_findings(strong, GATES) == []
    weak = "assert delta > 0\nassert outcome['gate_cleared'] is True\n"
    findings = weak_assertion_findings(weak, GATES)
    assert any("never asserts the recorded value 0.002158" in f
               for f in findings)


def test_staged_output_trips_actually_invoked():
    clean = actually_invoked_findings(
        ["inputs/train.parquet", "inputs/features.yaml"], GATES
    )
    assert clean == []
    leaked = actually_invoked_findings(
        ["inputs/train.parquet", "outputs/preds.csv"], GATES
    )
    assert any("freshly produced" in f for f in leaked)
