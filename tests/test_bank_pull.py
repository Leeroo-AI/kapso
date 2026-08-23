# Bank tool-surface tests (serving v2): eligibility law on the
# session-facing side — the index is the whole eligible set, quarantine
# refuses unmarked, refusals are named, and every event lands in the JSONL
# pull log (Rule 9: the regressions are a decoy leaking through direct
# request, silent refusals, and the gate resolving without its campaign
# env).

import json

from kapso.gated_mcp.presets import resolve_gates
from kapso.learning.bank import Bank
from kapso.learning.retriever import (
    append_pull_event,
    render_cards,
    render_index,
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


def test_index_is_the_whole_eligible_set(tmp_path):
    bank = make_pull_bank(tmp_path)
    result = render_index(bank, TASK)
    names = [row["card"] for row in result["listed"]]
    # out-of-scope, retired-state, and decoy cards never appear
    assert "avito-card" not in names
    assert "retired-state-card" not in names
    assert "decoy-card" not in names
    # the whole eligible set is listed
    assert {"strong-card", "weak-card", "tension-a", "tension-b"} <= set(names)
    # reliability order: strong before weak
    assert names.index("strong-card") < names.index("weak-card")
    assert all(row["exposure"] == "indexed" for row in result["listed"])


def test_reads_refuse_by_name_and_keep_decoys_unmarked(tmp_path):
    bank = make_pull_bank(tmp_path)
    result = render_cards(
        bank, TASK,
        ["strong-card", "avito-card", "decoy-card", "no-such-card"],
        False, {},
    )
    got_names = [row["card"] for row in result["served"]]
    assert got_names == ["strong-card"]
    assert all(row["exposure"] == "read" for row in result["served"])
    reasons = {r["card"]: r["reason"] for r in result["refused"]}
    assert "out of scope" in reasons["avito-card"]
    # a decoy refuses exactly like a nonexistent card — quarantine unmarked
    assert reasons["decoy-card"] == reasons["no-such-card"] == "no such card"
    # the full card body is rendered for the got card (Rule 6: never
    # clipped), title-first per format v2, with the citation tag line
    assert "[card:strong-card]" in result["text"]
    assert "# Use group-relative signals when rows compete in a pool" in result["text"]
    assert "**Confidence:**" in result["text"]


def test_pull_events_append_as_jsonl(tmp_path):
    log = tmp_path / "campaign" / "serving-pull.jsonl"
    append_pull_event(str(log), {"tool": "bank_index", "eligible": 4})
    append_pull_event(str(log), {"tool": "bank_get_card", "served": []})
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [row["tool"] for row in rows] == ["bank_index", "bank_get_card"]
    assert all("ts" in row for row in rows)


def test_bank_gate_resolves_only_with_campaign_env():
    resolution = resolve_gates(["bank"], policy="skip", env={})
    assert resolution.enabled_gates == ()
    missing = resolution.diagnostics[0].missing_env
    assert "KAPSO_BANK_DIR" in missing and "KAPSO_SERVING_PULL_LOG" in missing
    assert "KAPSO_PROBE_BUDGET" in missing
    resolution = resolve_gates(["bank"], policy="skip", env={
        "KAPSO_BANK_DIR": "/tmp/bank",
        "KAPSO_BANK_HEAD": "abc",
        "KAPSO_SERVING_PULL_LOG": "/tmp/pull.jsonl",
        "KAPSO_TASK_FAMILY": "entity_binary_classification",
        "KAPSO_PROBE_BUDGET": "1",
    })
    assert resolution.enabled_gates == ("bank",)
