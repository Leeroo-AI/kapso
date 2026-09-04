"""The bait suite as a live test (plan §3.3): every fixture, on the CLI
under test, `--bait-runs` times; one JSONL row per run under
tests/live/inbox_bait/results/. Skipped without --run-live.

The per-run assertion is the fixture's expectation (a benign trap draws
no request; a real blocker draws one with the right key); the suite-level
thresholds are read off the results by `inbox_bait.py report` and judged
by a person, as the plan says.
"""

import json
from pathlib import Path

import pytest

from inbox_bait import BAITS, build_bait, judge_tried, run_bait

RESULTS = Path(__file__).resolve().parent / "inbox_bait" / "results"
pytestmark = pytest.mark.live


def pytest_generate_tests(metafunc):
    if "bait_run" in metafunc.fixturenames:
        runs = int(metafunc.config.getoption("--bait-runs"))
        cli = metafunc.config.getoption("--bait-cli")
        metafunc.parametrize(
            "bait_run",
            [(name, cli, run) for run in range(1, runs + 1) for name in sorted(BAITS)],
            ids=[f"{name}-{cli}-{run}" for run in range(1, runs + 1) for name in sorted(BAITS)],
        )


def test_bait(bait_run, tmp_path):
    name, cli, run = bait_run
    root = tmp_path / f"{name}-{cli}-{run}"
    build_bait(root, name, cli)
    result = run_bait(root)
    result["run"] = run
    if result.get("requests"):
        judge_tried(root)
        result = json.loads((root / "result.json").read_text())
        result["run"] = run
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / f"{cli}.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(result) + "\n")
    assert result["verdict"] == "pass", json.dumps(
        {k: result.get(k) for k in ("trap", "requested", "requests", "score", "last_stop")}, indent=2
    )
