# Codify reproduction gates (CD§2) — the mechanical half of "PASS =
# mechanical gates green AND judge endorsement".
#
# The gates come from the codifier's request, drawn from the fixture run's
# recorded outcomes; the replay evaluation writes an outcome file the frame
# judges here. Decision outcomes are exact; numeric outcomes reproduce
# within ±tolerance_z·SE of the recorded value; artifact outcomes are
# property-checked. Two inherited checks: anti-weak-test (the evaluation
# must assert the recorded values, not something weaker) and
# actually-invoked (the workspace was staged with fixture inputs only, so
# every gated artifact must be freshly produced).

from pathlib import Path
from typing import Any, Dict, List


def reproduction_findings(
    gates: Dict[str, Any],
    observed: Dict[str, Any],
    workspace: str,
    tolerance_z: float,
) -> List[str]:
    """Judge one replay run's outcome against the gates. Every miss is a
    named finding; empty means mechanically green.

    gates: {decisions: {name: expected}, metrics: {name: {value, se}},
            artifacts: {name: {path, min_bytes?}}}
    observed: {decisions: {name: value}, metrics: {name: value}} — written
    by the replay evaluation itself."""
    findings: List[str] = []
    observed_decisions = observed.get("decisions") or {}
    for name, expected in (gates.get("decisions") or {}).items():
        if name not in observed_decisions:
            findings.append(f"decision {name}: not reported by the replay")
        elif observed_decisions[name] != expected:
            findings.append(
                f"decision {name}: expected {expected!r}, replay produced "
                f"{observed_decisions[name]!r} — decision outcomes are exact"
            )
    observed_metrics = observed.get("metrics") or {}
    for name, spec in (gates.get("metrics") or {}).items():
        value = float(spec["value"])
        se = float(spec["se"])
        if name not in observed_metrics:
            findings.append(f"metric {name}: not reported by the replay")
            continue
        produced = float(observed_metrics[name])
        band = tolerance_z * se
        if abs(produced - value) > band:
            findings.append(
                f"metric {name}: {produced} is outside {value} ± "
                f"{tolerance_z}·{se} — not a reproduction"
            )
    root = Path(workspace)
    for name, spec in (gates.get("artifacts") or {}).items():
        target = root / str(spec["path"])
        if not target.is_file():
            findings.append(f"artifact {name}: {spec['path']} was not produced")
            continue
        min_bytes = spec.get("min_bytes")
        if min_bytes is not None and target.stat().st_size < int(min_bytes):
            findings.append(
                f"artifact {name}: {spec['path']} is "
                f"{target.stat().st_size} bytes, below the recorded "
                f"{min_bytes} — a stub, not the outcome"
            )
    return findings


def weak_assertion_findings(
    replay_source: str, gates: Dict[str, Any]
) -> List[str]:
    """Anti-weak-test (CD§2): the replay evaluation must assert the card's
    recorded outcomes — mechanically, every gated reference value must
    appear in the evaluation source. A green run whose assertions do not
    encode the recorded values is weaker than `expected_outcome`."""
    findings: List[str] = []
    for name, spec in (gates.get("metrics") or {}).items():
        reference = str(spec["value"])
        if reference not in replay_source:
            findings.append(
                f"weak assertion: the replay evaluation never asserts the "
                f"recorded value {reference} for metric {name}"
            )
    for name, expected in (gates.get("decisions") or {}).items():
        if str(name) not in replay_source:
            findings.append(
                f"weak assertion: the replay evaluation never checks "
                f"decision {name}"
            )
    return findings


def actually_invoked_findings(
    staged_paths: List[str], gates: Dict[str, Any]
) -> List[str]:
    """Actually-invoked (CD§2): the workspace is staged with fixture INPUTS
    only — a gated artifact already present at staging is a leaked output,
    and its later 'reproduction' proves nothing."""
    staged = {str(Path(path)) for path in staged_paths}
    findings: List[str] = []
    for name, spec in (gates.get("artifacts") or {}).items():
        if str(Path(str(spec["path"]))) in staged:
            findings.append(
                f"artifact {name}: {spec['path']} was staged into the "
                f"workspace — outputs must be freshly produced"
            )
    return findings
