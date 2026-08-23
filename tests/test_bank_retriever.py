# Bank read model + retriever serving-v2 core tests.
#
# Each test names the regression it catches (Rule 9). Fixtures are hand-built
# bank checkouts under tmp_path — the schema is the design §3.2 card; the
# serving surface is serving-agentic-redesign.md (intro + index + cards at
# two depths).

from pathlib import Path

import pytest

from kapso.learning.bank import Bank, Card
from kapso.learning.retriever import (
    compile_intro,
    probe_offers,
    render_cards,
    render_index,
)

TASK = {"family": "entity_binary_classification", "dataset": "rel-hm"}


# One shared confidence string: the assessor writes it into BOTH
# reliability.plain and the body's **Confidence:** line (sync-checked).
PLAIN_CONFIDENCE = ("promising — confirmed in 1 campaign on 1 dataset; "
                    "no counter-evidence; untested elsewhere.")

TEMPLATE_BODY = (
    "# Use group-relative signals when rows compete in a pool\n\n"
    "**Rule:** When ranked rows compete inside a shared pool and your model "
    "consumes absolute per-row features, build group-relative features — "
    "within-group percentiles and z-scores — and feed them alongside the "
    "raw columns, because competition happens within the pool and absolute "
    "magnitudes hide the ordering signal; where rows do not compete, keep "
    "the absolute view and spend the slot elsewhere.\n\n"
    "## Is this your situation?\n\n"
    "- You are ranking rows that compete inside a shared pool.\n"
    "- Your model consumes absolute per-row features today.\n"
    "- You are choosing which normalization block to build next before "
    "training the next candidate.\n\n"
    "## What to do\n\n"
    "1. Group rows by their competing pool.\n"
    "2. Compute each feature's percentile or z-score within its group.\n"
    "3. Feed those transforms alongside the raw columns.\n"
    "4. Gate the block with your usual paired significance check before "
    "shipping the change into the current best model.\n\n"
    "## Why believe this\n\n"
    "Competition happens within the pool rather than across it, so "
    "absolute magnitudes mislead exactly when they look informative. In "
    "our runs the relative block separated competing rows at the margin "
    "where the absolute view ranked them identically, and it shipped "
    "after clearing the paired gate.\n\n"
    f"**Confidence:** {PLAIN_CONFIDENCE}"
)


def card_text(
    name,
    kind="insight",
    scope="domain",
    score=0.7,
    state="active",
    tags=(),
    contradicts=(),
    evidence_trajectory="rel-amazon--user-churn/20260101T000000_lane-t1",
    body=TEMPLATE_BODY,
):
    tags_yaml = "[" + ", ".join(tags) + "]"
    contradicts_yaml = "[" + ", ".join(contradicts) + "]"
    scope_yaml = scope if isinstance(scope, str) else "[" + ", ".join(scope) + "]"
    return f"""{body}

---

```yaml
type: {kind}
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
  plain: {PLAIN_CONFIDENCE}
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
probe: >-
  Ablate the grouped-rank block on one forward fold; keep the clustered delta.
```
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


def test_intro_and_launch_record(tmp_path):
    # Regression: the intro is the whole push surface — head stamp + the
    # three tool names, never card content; the launch record is v2 shape.
    root = build_bank(tmp_path, {"a-card": card_text("a-card")})
    result = compile_intro(Bank(str(root)), TASK, "lr_head")
    assert "lr_head" in result["intro"]
    for tool in ("bank_index()", "bank_get_card(", "bank_get_card_with_evidence("):
        assert tool in result["intro"]
    assert "[card:a-card]" not in result["intro"]  # no card content pushed
    assert result["record"] == {
        "mode": "agentic", "task": dict(TASK), "bank_head": "lr_head",
        "gaps": ["no procedure in the bank covers this task's scope"],
    }


def test_index_is_whole_set_score_ordered_no_discount(tmp_path):
    # Regression: the index carries EVERY eligible card in plain reliability
    # order — no k cut, no visited discount, no visited/probe markers; each
    # card is name + hero + applies-when.
    root = build_bank(tmp_path, {
        "visited-card": card_text(
            "visited-card", score=0.6,
            evidence_trajectory="rel-hm--user-churn/20260101T000000_lane-t1"),
        "unvisited-card": card_text("unvisited-card", score=0.9),
        "third-card": card_text("third-card", score=0.2),
    })
    result = render_index(Bank(str(root)), TASK)
    listed = [row["card"] for row in result["listed"]]
    # raw score order — the 0.9 card leads regardless of dataset history
    assert listed == ["unvisited-card", "visited-card", "third-card"]
    assert all(row["exposure"] == "indexed" for row in result["listed"])
    text = result["text"]
    assert "[card:unvisited-card] score 0.9" in text
    assert "applies-when: rows share a competing group" in text
    assert "visited-here" not in text and "PROBE" not in text
    # the empty procedure shelf is a stated gap, not silence
    assert "gaps: no procedure in the bank covers this task's scope" in text


def test_index_purity_and_section_filter(tmp_path):
    # Regression: the index is a pure function of (task, checkout) — the
    # hindcast replays it byte-identical; the section filter subsets it.
    root = build_bank(tmp_path, {
        "a-card": card_text("a-card"),
        "proc-card": card_text("proc-card", kind="procedure"),
    })
    bank = Bank(str(root))
    assert render_index(bank, TASK) == render_index(bank, TASK)
    full = render_index(bank, TASK)
    assert "## Insights" in full["text"] and "## Procedures" in full["text"]
    only_proc = render_index(bank, TASK, section="procedures")
    assert [row["card"] for row in only_proc["listed"]] == ["proc-card"]
    with pytest.raises(ValueError, match="unknown index section"):
        render_index(bank, TASK, section="pitfalls")


def test_render_cards_two_depths_and_rule6(tmp_path):
    # Regression: read depth is the whole body and nothing else; evidence
    # depth adds the reliability block + full evidence trail; the body is
    # never clipped at either depth (Rule 6).
    long_body = TEMPLATE_BODY.replace(
        "## Why believe this",
        "A fact stated at length. " * 200 + "\n\n## Why believe this",
    )
    root = build_bank(tmp_path, {"long-card": card_text("long-card", body=long_body)})
    bank = Bank(str(root))
    read = render_cards(bank, TASK, ["long-card"], False, {})
    assert "A fact stated at length. " * 200 in read["text"]
    assert "### Reliability" not in read["text"]
    assert read["served"][0]["exposure"] == "read"
    deep = render_cards(bank, TASK, ["long-card"], True, {})
    assert "### Reliability" in deep["text"]
    assert "### Evidence (1 entries)" in deep["text"]
    assert "verdict=confirm" in deep["text"]
    assert "KEPT at +0.0032" in deep["text"]
    assert deep["served"][0]["exposure"] == "evidence-read"


def test_probe_offers_cap_eligibility_and_ride_reads(tmp_path):
    # Regression: at most probe_budget offers, eligible cards only, and the
    # offer (with its cost clause) rides ONLY the offered card's read.
    from kapso.learning.retriever import compile_probe_queue

    root = build_bank(tmp_path, {
        "strong-card": card_text("strong-card", score=0.8),
        "weak-card": card_text("weak-card", score=0.4),
        "avito-card": card_text("avito-card", scope=["dataset:rel-avito"]),
    })
    bank = Bank(str(root))
    (root / "index").mkdir(exist_ok=True)
    (root / "index" / "probe-queue.md").write_text(compile_probe_queue(bank))
    offers = probe_offers(bank, TASK, 1)
    assert len(offers) == 1 and "avito-card" not in offers
    assert probe_offers(bank, TASK, 0) == {}
    offered_name = next(iter(offers))
    other = "weak-card" if offered_name != "weak-card" else "strong-card"
    with_offer = render_cards(bank, TASK, [offered_name], False, offers)
    assert "*probe:*" in with_offer["text"]
    assert "optional measurement offer" in with_offer["text"]
    without = render_cards(bank, TASK, [other], False, offers)
    assert "*probe:*" not in without["text"]


def test_procedure_read_carries_code_location(tmp_path):
    # Regression: a code-flipped procedure's read includes its code dir,
    # entrypoint, and replay dir; a prose-only procedure includes neither.
    flipped = card_text("flipped-proc", kind="procedure").replace(
        "provenance: {version: 1}",
        "provenance: {version: 1}\nentrypoint: code/main.py",
    )
    root = build_bank(tmp_path, {
        "flipped-proc": flipped,
        "prose-proc": card_text("prose-proc", kind="procedure"),
    })
    code_dir = root / "procedures" / "flipped-proc" / "code"
    code_dir.mkdir()
    (code_dir / "main.py").write_text("print('run')\n")
    (root / "procedures" / "flipped-proc" / "replay").mkdir()
    bank = Bank(str(root))
    result = render_cards(bank, TASK, ["flipped-proc", "prose-proc"], False, {})
    text = result["text"]
    assert f"code: {code_dir}" in text
    assert "entrypoint: " in text and "code/main.py" in text
    assert f"replay: {root / 'procedures' / 'flipped-proc' / 'replay'}" in text
    prose_section = text.split("[card:prose-proc]")[1]
    assert "code:" not in prose_section and "replay:" not in prose_section


def test_co_serving_guard_names_tension_on_reads(tmp_path):
    # Regression: a contradicts pair read together must name the tension.
    root = build_bank(tmp_path, {
        "card-a": card_text("card-a", contradicts=("card-b",)),
        "card-b": card_text("card-b", contradicts=("card-a",)),
    })
    result = render_cards(Bank(str(root)), TASK, ["card-a", "card-b"], False, {})
    assert result["tensions"] == [["card-a", "card-b"]]
    assert "treat as contested" in result["text"]


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


def test_probe_queue_ranks_by_voi_tiers(tmp_path):
    # P6 regression: the queue is ledger-derived - heavily-served thinly-
    # verified first (uncertainty x exposure), contradicts pairs as boundary
    # rows, zero-outcome candidates as blocked; decoys and probe-less cards
    # never appear.
    from kapso.learning.retriever import compile_probe_queue

    served_thin = card_text("served-thin", score=0.5)
    served_thin = served_thin.replace("validity: 0.8", "validity: 0.3")
    served_thin = served_thin.replace(
        "usage: independent evidence.", "usage: served and cited by the spec."
    )
    root = build_bank(tmp_path, {
        "served-thin": served_thin,
        "tension-a": card_text("tension-a", contradicts=("tension-b",)),
        "tension-b": card_text("tension-b", contradicts=("tension-a",)),
        "blocked-card": card_text("blocked-card", state="candidate").replace(
            "verdict: confirm", "verdict: exercise"
        ),
        "probeless": card_text("probeless").replace(
            "probe: >-\n  Ablate the grouped-rank block on one forward fold; "
            "keep the clustered delta.",
            "probe: null",
        ),
        "decoy-card": card_text("decoy-card"),
    })
    (root / ".decoys.yaml").write_text("- decoy-card\n")
    queue = compile_probe_queue(Bank(str(root)))
    lines = [l for l in queue.splitlines() if l and l[0].isdigit()]
    assert "[card:served-thin]" in lines[0] and "served-unverified" in lines[0]
    assert "voi=0.70" in lines[0]  # (1-0.3) x 1 exposure
    boundary = [l for l in lines if "boundary" in l]
    assert any("tension-a" in l for l in boundary)
    blocked = [l for l in lines if "blocked" in l]
    assert any("blocked-card" in l for l in blocked)
    assert "decoy-card" not in queue
    assert "probeless" not in " ".join(lines)

