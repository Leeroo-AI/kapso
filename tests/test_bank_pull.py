# Pull-tool core tests (P5.1): eligibility law on the session-facing side —
# the shortlist is the whole eligible set (query never filters), quarantine
# refuses unmarked, refusals are named, the co-serving guard runs on gets,
# and every event lands in the JSONL pull log (Rule 9: the regressions are a
# decoy leaking through direct request, silent refusals, and a padded list).

import json

from kapso.gated_mcp.presets import resolve_gates
from kapso.learning.bank import Bank
from kapso.learning.retriever import (
    append_pull_event,
    pull_projections,
    pull_shortlist,
)
from tests.test_bank_retriever import build_bank, card_text

TASK = {"family": "entity_binary_classification", "dataset": "rel-hm"}


def make_pull_bank(tmp_path):
    root = build_bank(tmp_path, {
        "strong-card": card_text("strong-card", score=0.8),
        "weak-card": card_text("weak-card", score=0.4),
        "avito-card": card_text("avito-card", scope=["dataset:rel-avito"]),
        "retired-state-card": card_text("retired-state-card", state="retired"),
        "decoy-card": card_text("decoy-card", score=0.9),
        "tension-a": card_text("tension-a", contradicts=("tension-b",)),
        "tension-b": card_text("tension-b", contradicts=("tension-a",)),
    })
    (root / ".decoys.yaml").write_text("- decoy-card\n")
    return Bank(str(root))


def test_shortlist_is_the_whole_eligible_set_query_never_filters(tmp_path):
    bank = make_pull_bank(tmp_path)
    result = pull_shortlist(bank, TASK, "anything about optimizers at all")
    names = [row["card"] for row in result["shown"]]
    # out-of-scope, retired-state, and decoy cards never appear
    assert "avito-card" not in names
    assert "retired-state-card" not in names
    assert "decoy-card" not in names
    # the whole eligible set is shown regardless of the query's words
    assert {"strong-card", "weak-card", "tension-a", "tension-b"} <= set(names)
    # reliability order: strong before weak
    assert names.index("strong-card") < names.index("weak-card")
    # exposure level is searched, and the census line is present
    assert all(row["exposure"] == "searched" for row in result["shown"])
    assert f"Eligible set: {len(names)}" in result["text"]


def test_projections_refuse_by_name_and_keep_decoys_unmarked(tmp_path):
    bank = make_pull_bank(tmp_path)
    result = pull_projections(
        bank, TASK,
        ["strong-card", "avito-card", "decoy-card", "no-such-card"],
    )
    got_names = [row["card"] for row in result["got"]]
    assert got_names == ["strong-card"]
    assert all(row["exposure"] == "got" for row in result["got"])
    reasons = {r["card"]: r["reason"] for r in result["refused"]}
    assert "out of scope" in reasons["avito-card"]
    # a decoy refuses exactly like a nonexistent card — quarantine unmarked
    assert reasons["decoy-card"] == reasons["no-such-card"] == "no such card"
    # the full card body is rendered for the got card (Rule 6: never
    # clipped), title-first per format v2, with the citation tag line
    assert "[card:strong-card]" in result["text"]
    assert "# Use group-relative signals when rows compete in a pool" in result["text"]
    assert "**Confidence:**" in result["text"]


def test_co_serving_guard_names_the_tension_on_gets(tmp_path):
    bank = make_pull_bank(tmp_path)
    result = pull_projections(bank, TASK, ["tension-a", "tension-b"])
    assert result["tensions"] == [["tension-a", "tension-b"]]
    assert "Contested:" in result["text"]


def test_pull_events_append_as_jsonl(tmp_path):
    log = tmp_path / "campaign" / "serving-pull.jsonl"
    append_pull_event(str(log), {"tool": "bank_search", "eligible": 4})
    append_pull_event(str(log), {"tool": "bank_get", "got": []})
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [row["tool"] for row in rows] == ["bank_search", "bank_get"]
    assert all("ts" in row for row in rows)


def test_bank_gate_resolves_only_with_campaign_env():
    resolution = resolve_gates(["bank"], policy="skip", env={})
    assert resolution.enabled_gates == ()
    missing = resolution.diagnostics[0].missing_env
    assert "KAPSO_BANK_DIR" in missing and "KAPSO_SERVING_PULL_LOG" in missing
    resolution = resolve_gates(["bank"], policy="skip", env={
        "KAPSO_BANK_DIR": "/tmp/bank",
        "KAPSO_BANK_HEAD": "abc",
        "KAPSO_SERVING_PULL_LOG": "/tmp/pull.jsonl",
        "KAPSO_TASK_FAMILY": "entity_binary_classification",
    })
    assert resolution.enabled_gates == ("bank",)
