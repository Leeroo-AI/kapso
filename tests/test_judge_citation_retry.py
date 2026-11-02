"""The judge's one re-ask on a citation slip (docs/plans/redesign/
wp9-semantic-suite.md, S2's round): a reply whose quote is not on the line
it cites, everything else in shape, is asked once more, both raw replies
kept; a second slip fails loud as any other contract break; any other
fault is left to validate to name."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("judge", ROOT / "scripts" / "redesign" / "judge.py")
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)
DIMENSIONS = judge.load_suite()["tests"]["S2"]["judge"]["dimensions"]
WATCH = "judge/watch.rendered.md"


def _reply(line: str) -> str:
    return json.dumps({"scores": {d: {"score": 4, "evidence": [{"where": f"{WATCH}:{line}", "quote": "no score yet"}]} for d in DIMENSIONS}})


def test_a_quote_on_the_wrong_line_is_asked_once_more_then_fails_loud(tmp_path, monkeypatch):
    # Two of ten judge runs on 2026-10-28 lost their report to a citation
    # the model got wrong by a line.
    rendered = tmp_path / "judge" / "watch.rendered.md"
    rendered.parent.mkdir()
    rendered.write_text("L1: # step watch\nL2: \nL3: [assistant]\nL4: Running, iteration 1 in ideation; no score yet.\n")
    # Two lines off: one line off is read as the citation (rel-s4-codex).
    slipped, right = _reply("L2"), _reply("L4")
    replies = iter([slipped, right])
    monkeypatch.setattr(judge, "ask", lambda prompt, settings: next(replies))
    scores = judge.judge_material("material", "rubric", DIMENSIONS, [1, 5], {}, tmp_path, rendered.parent, "watch-pass1", [WATCH])
    assert scores["clear"]["score"] == 4
    assert (rendered.parent / "raw-watch-pass1.txt").read_text() == slipped
    assert (rendered.parent / "raw-watch-pass1-retry.txt").read_text() == right
    replies = iter([slipped, slipped])
    with pytest.raises(judge.JudgeFailure, match="is not on"):
        judge.judge_material("material", "rubric", DIMENSIONS, [1, 5], {}, tmp_path, rendered.parent, "watch-pass2", [WATCH])
    # A sentence the rendering wrapped is cited by the line it starts on.
    rendered.write_text("L1: # step watch\nL2: \nL3: [assistant]\nL4: Running, iteration 1 in ideation;\nL5: no score yet.\n")
    lines = rendered.read_text().splitlines()
    assert judge.quote_on(lines, "4", "Running, iteration 1 in ideation; no score yet.")
    assert judge.quote_on(lines, "4", "iteration 1 in ideation;")
    assert not judge.quote_on(lines, "2", "Running, iteration 1 in ideation; no score yet."), "two lines off is not the citation"
    assert not judge.quote_on(lines, "4", "the best score is 0.9")
    # The words, not the markdown around them: a bullet and emphasis folded on both sides.
    marked = ["L10: - **Data:** `data/transactions.csv` has 20,000 labelled rows, `fraud_probability` in input order."]
    assert judge.quote_on(marked, "1", "Data: `data/transactions.csv` has 20,000 labelled rows, `fraud_probability` in input order.")
    assert judge.quote_on(marked, "1", "**Data:** `data/transactions.csv` has 20,000")
    assert not judge.quote_on(marked, "1", "Data: `data/transactions.csv` has 30,000")
    assert judge.quote_on(marked, "1", "Data: data/transactions.csv has 20,000 labelled rows, fraud_probability in input order.")
    assert judge.quote_on(["L3: ### What's here"], "1", "What's here")
    # A fault that is not a misquote is validate's to name, never retried.
    assert not judge.misquoted({"scores": {"clear": {"score": 4, "evidence": []}}}, tmp_path)
    assert not judge.misquoted({"scores": {"clear": {"score": 4, "evidence": [{"where": "judge/elsewhere.md:L1", "quote": "x"}]}}}, tmp_path)


def test_a_conversation_step_with_no_exchange_is_skipped_but_a_missing_transcript_still_raises(tmp_path):
    # s2-codex-5: harness-export on the Codex host ran no session (zero
    # messages, no transcript) and the judge died on it; a step with
    # messages and no transcript is still a broken record.
    (tmp_path / "transcripts").mkdir()
    archive = tmp_path / "judge"
    archive.mkdir()
    record = {"steps": [{"name": "harness-export", "kind": "conversation", "messages": 0}]}
    assert judge.gather_materials(tmp_path, record, None, archive) == {}
    record = {"steps": [{"name": "watch", "kind": "conversation", "messages": 2}]}
    with pytest.raises(judge.JudgeFailure, match="no transcript"):
        judge.gather_materials(tmp_path, record, None, archive)
