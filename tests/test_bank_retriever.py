# Bank read model + retriever push core tests (P3).
#
# Each test names the regression it catches (Rule 9). Fixtures are hand-built
# bank checkouts under tmp_path — the schema is the design §3.2 card.

from pathlib import Path

import pytest

from kapso.learning.bank import Bank, Card
from kapso.learning.retriever import compile_brief

RETRIEVER_CONFIG = {
    "k_insights": 2,
    "k_procedures": 1,
    "k_pitfalls": 1,
    "unvisited_discount": 0.5,
}

TASK = {"family": "entity_binary_classification", "dataset": "rel-hm"}


def card_text(
    name,
    kind="insight",
    scope="domain",
    score=0.7,
    state="active",
    tags=(),
    contradicts=(),
    evidence_trajectory="rel-amazon--user-churn/20260101T000000_lane-t1",
    body="Group-relative signals beat absolute values [E1]. The mechanism is "
         "competition within a shared pool.",
):
    tags_yaml = "[" + ", ".join(tags) + "]"
    contradicts_yaml = "[" + ", ".join(contradicts) + "]"
    scope_yaml = scope if isinstance(scope, str) else "[" + ", ".join(scope) + "]"
    return f"""---
type: {kind}
title: {name.replace('-', ' ').title()}
description: >-
  A hero line for {name}.
tags: {tags_yaml}
timestamp: 2026-08-14T09:00:00Z
scope: {scope_yaml}
scope_conditions: "rows share a competing group"
evidence:
  - source:
      learner_run: lr_20260814T0900
      trajectory: {evidence_trajectory}
      ref: campaign.log
      card_version: null
    verdict: confirm
    usage: independent evidence.
    effect: KEPT at +0.0032 (~3.6 SE) — significant in-scope agreement.
reliability:
  validity: 0.8
  boundary: 0.5
  coverage: 0.3
  score: {score}
  rationale: >-
    Validity from two confirmations; boundary untested; coverage thin.
  state: {state}
provenance: {{version: 1}}
log:
  - version: 1
    date: 2026-08-14
    commit: lr_20260814T0900
    change: Created from two independent instances.
supersedes: null
contradicts: {contradicts_yaml}
---
{body}
"""


def build_bank(tmp_path, cards=None):
    root = tmp_path / "bank"
    (root / "insights").mkdir(parents=True)
    (root / "procedures").mkdir()
    cards = cards if cards is not None else {}
    names = {"insights": [], "procedures": []}
    for name, text in cards.items():
        if "type: procedure" in text:
            (root / "procedures" / name).mkdir()
            (root / "procedures" / name / "card.md").write_text(text)
            names["procedures"].append(name)
        else:
            (root / "insights" / f"{name}.md").write_text(text)
            names["insights"].append(name)
    for section, listed in names.items():
        (root / section / "index.md").write_text(
            "\n".join(f"- [{n}]({n}.md) — hero" for n in listed) or "empty"
        )
    return root


def test_card_parse_and_eligibility(tmp_path):
    # Regression: scope is the eligibility law — domain covers everything, a
    # coordinate list covers exactly its coordinates, tags never participate.
    root = build_bank(tmp_path, {
        "domain-card": card_text("domain-card", scope="domain"),
        "hm-card": card_text("hm-card", scope=["dataset:rel-hm"]),
        "family-card": card_text(
            "family-card", scope=["family:entity_binary_classification"]),
        "other-card": card_text("other-card", scope=["dataset:rel-avito"],
                                tags=("data:grouped_rows",)),
    })
    bank = Bank(str(root))
    assert bank.cards["domain-card"].eligible_for(TASK)
    assert bank.cards["hm-card"].eligible_for(TASK)
    assert bank.cards["family-card"].eligible_for(TASK)
    assert not bank.cards["other-card"].eligible_for(TASK)


def test_quarantine_is_law(tmp_path):
    # Regression (§5.1): decoys and retired cards are unservable, whatever
    # their scores say.
    root = build_bank(tmp_path, {
        "good-card": card_text("good-card"),
        "retired-card": card_text("retired-card", state="retired"),
        "decoy-card": card_text("decoy-card", score=0.99),
    })
    (root / ".decoys.yaml").write_text("- decoy-card\n")
    bank = Bank(str(root))
    served = {c.name for c in bank.servable()}
    assert served == {"good-card"}


def test_push_brief_rank_and_record(tmp_path):
    # Regression: rank = reliability order with the unvisited discount; the
    # serving record carries versions and rank components; k caps selection.
    root = build_bank(tmp_path, {
        # visited (evidence trajectory on rel-hm) at 0.6 -> effective 0.6
        "visited-card": card_text(
            "visited-card", score=0.6,
            evidence_trajectory="rel-hm--user-churn/20260101T000000_lane-t1"),
        # unvisited at 0.9 -> effective 0.45, ranks BELOW the visited 0.6
        "unvisited-card": card_text("unvisited-card", score=0.9),
        "third-card": card_text("third-card", score=0.2),
    })
    result = compile_brief(Bank(str(root)), TASK, "lr_test", RETRIEVER_CONFIG)
    served = [s["card"] for s in result["record"]["served"]]
    assert served == ["visited-card", "unvisited-card"]  # k_insights=2 caps third
    assert result["record"]["served"][0]["visited"] is True
    assert result["record"]["served"][1]["effective"] == pytest.approx(0.45)
    assert result["record"]["bank_head"] == "lr_test"
    assert all(s["exposure"] == "got" for s in result["record"]["served"])
    # unvisited serving surfaces in the gap analysis
    assert any("unvisited-card" in g for g in result["record"]["gaps"])


def test_push_purity(tmp_path):
    # Regression: push is a pure function of (task, bank_head) — byte-equal
    # output on repeated calls (the hindcast replays it at historical heads).
    root = build_bank(tmp_path, {"a-card": card_text("a-card")})
    first = compile_brief(Bank(str(root)), TASK, "lr_x", RETRIEVER_CONFIG)
    second = compile_brief(Bank(str(root)), TASK, "lr_x", RETRIEVER_CONFIG)
    assert first == second


def test_co_serving_guard_names_tension(tmp_path):
    # Regression: a contradicts pair served together must name the tension.
    root = build_bank(tmp_path, {
        "card-a": card_text("card-a", contradicts=("card-b",)),
        "card-b": card_text("card-b", contradicts=("card-a",)),
    })
    result = compile_brief(Bank(str(root)), TASK, "lr_x", RETRIEVER_CONFIG)
    assert result["record"]["tensions"] == [["card-a", "card-b"]]
    assert "treat as contested" in result["brief"]


def test_pitfalls_ride_along_and_gaps_state_absences(tmp_path):
    # Regression: pitfall-tagged insights ride as guardrails on their own
    # budget; an empty procedure shelf is a stated gap, not silence.
    root = build_bank(tmp_path, {
        "pit-card": card_text("pit-card", tags=("pitfall",), score=0.1),
        "main-card": card_text("main-card"),
    })
    result = compile_brief(Bank(str(root)), TASK, "lr_x", RETRIEVER_CONFIG)
    served = [s["card"] for s in result["record"]["served"]]
    assert "pit-card" in served and "main-card" in served
    assert "Pitfall guardrails" in result["brief"]
    assert any("no procedure" in g for g in result["record"]["gaps"])


def test_full_fact_never_clipped(tmp_path):
    # Regression (Rule 6): k caps selection, never content — the body rides
    # whole into the brief.
    long_body = "A fact stated at length. " * 200
    root = build_bank(tmp_path, {"long-card": card_text("long-card", body=long_body)})
    result = compile_brief(Bank(str(root)), TASK, "lr_x", RETRIEVER_CONFIG)
    assert long_body.strip() in result["brief"]


def test_conformance_findings(tmp_path):
    # Regression: the OKF conformance gate — schema violations and index
    # omissions are named findings.
    good = card_text("good-card")
    bad = good.replace("scope: domain", "scope: []").replace(
        "  rationale: >-\n    Validity from two confirmations; boundary untested; coverage thin.\n",
        "  rationale: ''\n",
    )
    root = build_bank(tmp_path, {"good-card": good, "bad-card": bad})
    (root / "insights" / "index.md").write_text("- [good-card](good-card.md) — hero")
    findings = Bank(str(root)).conformance_findings()
    assert any("bad-card: scope" in f for f in findings)
    assert any("bad-card: reliability rationale missing" in f for f in findings)
    assert any("does not list bad-card" in f for f in findings)
    assert not any(f.startswith("good-card:") for f in findings)


def test_version_log_one_to_one(tmp_path):
    # Regression: provenance version ⇔ log entries, the invariant the write
    # side enforces and the read side reports.
    text = card_text("v-card").replace("provenance: {version: 1}",
                                       "provenance: {version: 3}")
    root = build_bank(tmp_path, {"v-card": text})
    findings = Bank(str(root)).conformance_findings()
    assert any("version ⇔ log-entry" in f for f in findings)
